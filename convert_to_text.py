import os
import ray
import warnings
from pathlib import Path
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
from ray.data import ActorPoolStrategy
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import openai
import time
import psycopg
from pgvector.psycopg import register_vector
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

warnings.filterwarnings("ignore")

DB_CONNECTION_STRING = "postgresql://localhost:5432/postgres?user=postgres"

# CREATE TABLE imagepdftext (id serial primary key, "text" text not null,
# source text not null, page_num INTEGER, page_img text not null, embedding vector(768))

import env

os.environ["OPENAI_API_KEY"] = env.OPENAI_APIKEY


class TokenizerAndAgent:

    def __init__(self):
        pass

    def reload_module(self):
        # Initialize cluster
        if ray.is_initialized():
            ray.shutdown()
        ray.init(
            runtime_env={
                "env_vars": {
                    "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                },
            }
        )
        ray.cluster_resources()

        ext_img = [".jpg", ".jpeg", ".png"]
        ext_pdf = [".pdf"]

        images_dir = Path("uploaded_data/")
        ds_images = ray.data.from_items(
            [{"path": path.as_posix()} for path in images_dir.rglob("*") if
             path.suffix in ext_img and not path.is_dir()]
        )
        print(f"{ds_images.count()} images")

        pdf_dir = Path("uploaded_data/")
        ds_pdf = ray.data.from_items(
            [{"path": path.as_posix()} for path in pdf_dir.rglob("*") if path.suffix in ext_pdf and not path.is_dir()]
        )
        print(f"{ds_pdf.count()} pdfs")

        class ExtractPagesFromPdf:
            def __init__(self):
                self.ocr_model = ocr_predictor(pretrained=True)

            def __call__(self, batch):
                sourcefile = str(batch["path"])[14:]
                print("Entered ExtractPagesFromPdf source: ", sourcefile)

                doc = DocumentFile.from_pdf(batch["path"])
                # the list of pages decoded as numpy ndarray of shape H x W x C
                return [{"page_content": doc[idx], "source": sourcefile, "page_number": idx} for idx in range(len(doc))]

        class ExtractPdfText:
            def __init__(self):
                self.ocr_model = ocr_predictor(pretrained=True)

            def __call__(self, batch):
                print("Entered ExtractPdfText pagenumber: ", batch["page_number"])

                text = self.ocr_model([batch["page_content"]])

                # img_pdf: np.ndarray = text.pages[0].synthesize()
                # size = text.pages[0].synthesize().size
                # new_image = Image.new(img_pdf, ((1440, 900), (255, 255, 255)))
                # img = Image(img_pdf, clamp=True)
                # , mode="L"
                img = Image.fromarray(np.array(batch["page_content"]))
                sourcefile = str(batch["source"])[:-4]
                current_datetime = str(datetime.now().strftime("%Y%m%d-%H%M%S%f"))
                image_file_name = sourcefile + "_" + current_datetime + ".jpg"
                path_to_save_image = "data_backup/" + image_file_name
                print("path_to_save_image :", path_to_save_image)
                img.save(path_to_save_image)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

                chunks = text_splitter.create_documents(
                    texts=[text.render()],
                    metadatas=[{"source": batch["source"]}])

                # chunks = text_splitter.split_documents([text])
                for chunk in chunks:
                    print("pdf page: ", chunk.page_content)

                return [{"text": chunk.page_content, "source": batch["source"],
                         "page_number": batch["page_number"], "page_image": path_to_save_image}
                        for chunk in chunks]

        class ExtractImageText:
            def __init__(self):
                self.ocr_model = ocr_predictor(pretrained=True)

            def __call__(self, batch):
                doc = DocumentFile.from_images(batch["path"])
                text = self.ocr_model(doc)

                sourcefile = str(batch["path"])[14:]
                print("img sourcefile :", sourcefile)
                path_to_save_image = "data_backup/" + sourcefile

                # print("image text: ", text.render())
                # return {"source": batch["path"], "text": [text.render()]}
                return [{"source": sourcefile, "text": text.render(), "page_number": 0, "page_image": path_to_save_image}]

        class EmbedTexts:
            def __init__(self, model_name="thenlper/gte-base"):
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"device": "cpu", "batch_size": 4},
                )

            def __call__(self, batch):
                embeddings = self.embedding_model.embed_documents(batch["text"])
                return {
                    "text": batch["text"],
                    "source": batch["source"],
                    "page_num": batch["page_number"],
                    "page_img": batch["page_image"],
                    "embeddings": embeddings,
                }

        class StoreEmbeddings:
            def __call__(self, batch):
                with psycopg.connect(DB_CONNECTION_STRING, password="postgres") as conn:
                    register_vector(conn)
                    with conn.cursor() as cur:
                        for text, source, page_num, page_img, embedding in zip(batch["text"],
                                                                               batch["source"], batch["page_num"],
                                                                               batch["page_img"], batch["embeddings"]):
                            cur.execute("INSERT INTO imagepdftext (text, source, page_num, page_img, embedding) "
                                        "VALUES (%s, %s, %s, %s, %s)",
                                        (text, source, int(page_num), page_img, embedding,), )
                return {}

        if ds_pdf.count():
            pdf_pages = ds_pdf.flat_map(ExtractPagesFromPdf, compute=ActorPoolStrategy(size=1))
            pdf_texts_ds = pdf_pages.flat_map(ExtractPdfText, compute=ActorPoolStrategy(size=1))

            # is_allow = False
            # if is_allow:
            pdf_embedded_texts = pdf_texts_ds.map_batches(
                EmbedTexts,
                fn_constructor_kwargs={"model_name": "thenlper/gte-base"},
                compute=ActorPoolStrategy(size=1),
            )

            # sample = pdf_embedded_texts.take(1)
            # print(f"pdf_embedded_texts embedding size: {len(sample[0]['embeddings'])}")

            pdf_embedded_texts.map_batches(
                StoreEmbeddings,
                batch_size=4,
                num_cpus=1,
                compute=ActorPoolStrategy(size=1),
            ).count()

        if ds_images.count():
            # Extract texts from images
            image_texts_ds = ds_images.flat_map(ExtractImageText, compute=ActorPoolStrategy(size=1))

            image_embedded_texts = image_texts_ds.map_batches(
                EmbedTexts,
                fn_constructor_kwargs={"model_name": "thenlper/gte-base"},
                compute=ActorPoolStrategy(size=1),
            )

            # sample = image_embedded_texts.take(1)
            # print(f"image_embedded_texts embedding size: {len(sample[0]['embeddings'])}")

            image_embedded_texts.map_batches(
                StoreEmbeddings,
                batch_size=4,
                num_cpus=1,
                compute=ActorPoolStrategy(size=1),
            ).count()

    # clear data from uploaded_data
    files_to_delete = os.listdir("uploaded_data/")
    for file in files_to_delete:
        file_path = os.path.join("uploaded_data/", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # ***************END OF INIT_MODULE***************************#

    def semantic_search(self, query, embedding_model, num_of_chunks):
        question_embedding = np.array(embedding_model.embed_query(query))
        with psycopg.connect(DB_CONNECTION_STRING, password="postgres") as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, source, text, page_num, page_img FROM imagepdftext ORDER BY embedding <=> %s LIMIT %s",
                    (question_embedding, num_of_chunks),
                )
                rows = cur.fetchall()
                semantic_context = [
                    {"id": row[0], "source": row[1], "text": row[2], "page_num": row[3], "page_img": row[4]} for row in
                    rows
                ]
        return semantic_context

    def generate_response(self,
                          llm,
                          temperature=0.0,
                          system_content="",
                          assistant_content="",
                          user_content="",
                          max_retries=1,
                          retry_interval=60,
                          ):
        """Generate response from an LLM."""
        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=llm,
                    temperature=temperature,
                    stream=False,
                    api_key=os.environ["OPENAI_API_KEY"],
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "assistant", "content": assistant_content},
                        {"role": "user", "content": user_content},
                    ],
                )
                return response["choices"][-1]["message"]["content"]
            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(retry_interval)
                retry_count += 1
        return ""

    def execute_query(self, query, num_chunks=1):
        # llm = "meta-llama/Llama-2-7b-chat-hf"
        llm = "gpt-4"
        embedding_model_name = "thenlper/gte-base"
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        temperature = 0.0
        max_context_length = 4096
        assistant_content = ""
        system_content = "Answer the query using the context provided. Be succinct."
        num_of_chunks = num_chunks

        # Context length (restrict input length to 50% of total length)
        # max_context_length = int(0.5 * max_context_length)
        # self.context_length = max_context_length - get_num_tokens(system_content + assistant_content)

        # Get sources and context
        context_results = self.semantic_search(query, embedding_model, num_of_chunks)

        # Generate response
        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        page_nums = [item["page_num"] for item in context_results]
        page_imgs = [item["page_img"] for item in context_results]
        user_content = f"query: {query}, context: {context}"

        answer = self.generate_response(llm=llm, temperature=temperature, system_content=system_content,
                                        assistant_content=assistant_content, user_content=user_content)

        # Result
        response = {
            "question": query,
            "sources": sources,
            "answer": answer,
            "page_num": page_nums,
            "page_img": page_imgs,
            "llm": llm,
            "context": context,
        }

        print("context: ", response.get("context"))
        print("sources: ", response.get("sources"))
        return response


# ***************END OF CLASS***************************#


# init_pdf_module(True)
# query = "whats the website mentioned to Get a quote for the completion of  template "
# use_agent(query)

# get_response_to_user_query()
# main()
# query_summarize()

def test_path():
    PDF_DIR = Path("data_backup/")
    # files = [p for p in Path(mainpath).rglob('*') if p.suffix in exts]
    exts = [".jpg", ".jpeg", ".png"]
    exts1 = [".pdfex"]
    # mainpath = "/path/to/dir"

    ds = ray.data.from_items(
        [{"path": path.as_posix()} for path in PDF_DIR.rglob("*") if path.suffix in exts1 and not path.is_dir()]
    )
    print(f"{ds.count()} images")

# reload_module()
# test_path()
