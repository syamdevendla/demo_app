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
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

DB_CONNECTION_STRING = "postgresql://localhost:5432/postgres?user=postgres"

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

        class ExtractPdfText:
            def __init__(self):
                pass

            def __call__(self, batch):
                sourcefilepath = str(batch["path"])
                sourcefile = sourcefilepath[14:]
                print("Entered PyPDFLoader 1 and str: ", sourcefile)
                loader = PyPDFLoader(str(batch["path"]), extract_images=True)
                data = loader.load()

                # if os.path.isfile(sourcefilepath):
                # os.remove(sourcefilepath)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                chunks = text_splitter.split_documents(data)
                return [{"text": chunk.page_content, "source": sourcefile} for chunk in chunks]

        class ExtractImageText:
            def __init__(self):
                self.ocr_model = ocr_predictor(pretrained=True)

            def __call__(self, batch):
                doc = DocumentFile.from_images(batch["path"])

                text = self.ocr_model(doc)
                print("image text: ", text.render())
                return {"source": batch["path"], "text": [text.render()]}

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
                    "embeddings": embeddings,
                }

        class StoreEmbeddings:
            def __call__(self, batch):
                with psycopg.connect(DB_CONNECTION_STRING, password="postgres") as conn:
                    register_vector(conn)
                    with conn.cursor() as cur:
                        for text, source, embedding in zip(batch["text"], batch["source"], batch["embeddings"]):
                            cur.execute("INSERT INTO infographic (text, source, embedding) VALUES (%s, %s, %s)",
                                        (text, source, embedding,), )
                return {}

        if ds_pdf.count():
            pdf_texts_ds = ds_pdf.flat_map(ExtractPdfText, compute=ActorPoolStrategy(size=1))

            pdf_embedded_texts = pdf_texts_ds.map_batches(
                EmbedTexts,
                fn_constructor_kwargs={"model_name": "thenlper/gte-base"},
                compute=ActorPoolStrategy(size=1),
            )

            sample = pdf_embedded_texts.take(1)
            print(f"pdf_embedded_texts embedding size: {len(sample[0]['embeddings'])}")

            pdf_embedded_texts.map_batches(
                StoreEmbeddings,
                batch_size=4,
                num_cpus=1,
                compute=ActorPoolStrategy(size=1),
            ).count()

        if ds_images.count():
            # Extract texts from images
            image_texts_ds = ds_images.map_batches(ExtractImageText, compute=ActorPoolStrategy(size=1))

            image_embedded_texts = image_texts_ds.map_batches(
                EmbedTexts,
                fn_constructor_kwargs={"model_name": "thenlper/gte-base"},
                compute=ActorPoolStrategy(size=1),
            )

            sample = image_embedded_texts.take(1)
            print(f"image_embedded_texts embedding size: {len(sample[0]['embeddings'])}")

            image_embedded_texts.map_batches(
                StoreEmbeddings,
                batch_size=4,
                num_cpus=1,
                compute=ActorPoolStrategy(size=1),
            ).count()

    # ***************END OF INIT_MODULE***************************#

    def semantic_search(self, query, embedding_model, num_of_chunks):
        question_embedding = np.array(embedding_model.embed_query(query))
        with psycopg.connect(DB_CONNECTION_STRING, password="postgres") as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, source, text FROM infographic ORDER BY embedding <=> %s LIMIT %s",
                    (question_embedding, num_of_chunks),
                )
                rows = cur.fetchall()
                semantic_context = [
                    {"id": row[0], "source": row[1], "text": row[2]} for row in rows
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
                # retry_count += 1
        return ""

    def execute_query(self, query):
        # llm = "meta-llama/Llama-2-7b-chat-hf"
        llm = "gpt-4"
        embedding_model_name = "thenlper/gte-base"
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        temperature = 0.0
        max_context_length = 4096
        assistant_content = ""
        system_content = "Answer the query using the context provided. Be succinct."
        num_of_chunks = 5

        # Context length (restrict input length to 50% of total length)
        # max_context_length = int(0.5 * max_context_length)
        # self.context_length = max_context_length - get_num_tokens(system_content + assistant_content)

        # Get sources and context
        context_results = self.semantic_search(query, embedding_model, num_of_chunks)

        # Generate response
        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"

        answer = self.generate_response(llm=llm, temperature=temperature, system_content=system_content,
                                        assistant_content=assistant_content, user_content=user_content)

        # Result
        response = {
            "question": query,
            "sources": sources,
            "answer": "",
            "llm": llm,
            "context": context,
        }

        print("context: ", response.get("context"))
        print("sources: ", response.get("sources"))
        return response.get("answer")


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
