# Pre-entrega 3 
Sistema RAG que responde preguntas sobre el Reglamento de Copropiedad de un edificio, usa solo la información contenida en esos documentos. Si la respuesta no está en el contexto recuperado, el modelo responde explícitamente que no la tiene.

## Arquitectura

1. **Ingesta**: `DirectoryLoader` carga los `.txt` de `data/` y `RecursiveCharacterTextSplitter`
   (con encoder de `tiktoken`) los fragmenta en chunks de **500 tokens con 50 de overlap**,
   usando separadores propios del documento (`\n⚬`, `\n\t`, `\n\n`, `\n`, `. `) para que
   cada norma empiece un chunk en vez de quedar como cola de otro tema. 
2. **Embeddings + persistencia**: los chunks se embeben con `HuggingFaceEmbeddings`
   (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, multilingüe: el corpus está
   en español) y se guardan en una colección de **ChromaDB** persistente. El índice se
   reutiliza entre ejecuciones y solo se reconstruye si cambiaron los documentos o la
   configuración de indexado.
3. **Retriever**: búsqueda por similitud (`k=4`) sobre la colección de Chroma.
4. **Generación grounded (LCEL)**: una única cadena que compone
   `RunnableParallel(retriever, pregunta) | formateo de documentos | prompt | ChatOpenAI |
   PydanticOutputParser`. El retriever vive **dentro** de la cadena, así los fragmentos
   recuperados alimentan a la vez el CONTEXTO del prompt y las fuentes citadas en la salida.
   El prompt de sistema obliga al modelo a responder solo con ese CONTEXTO.
5. **Salida validada con Pydantic**: la respuesta final se devuelve como `RAGRespuesta`
   (`respuesta`, `fuentes`, `fragmentos_recuperados`).

Todo el flujo se expone mediante la función asíncrona `get_rag_response(query: str)`, que
hace `await rag_chain.ainvoke(query)`: `ainvoke` se propaga por toda la cadena, de modo que
la búsqueda en Chroma y la llamada al LLM usan sus caminos asíncronos nativos.

## Requisitos

- Python 3.11+
- Una API key de OpenAI

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

Crear un archivo `.env` en la raíz del proyecto con:

```
OPENAI_API_KEY="tu-api-key-de-openai"
```

`.env` está en `.gitignore`: nunca subas tu API key al repositorio.

## Ejecución

Abrir `main.ipynb` y ejecutar las celdas en orden.

## Estructura del repo

```
main.ipynb           # notebook con todo el pipeline RAG
data/                # dataset de ejemplo (reglamento de copropiedad)
requirements.txt     # dependencias con versiones fijas
.env                 # credenciales locales 
vectorstore/         # índice de Chroma generado localmente 
```

## Notas

El script de ingesta está incluido en una celda del notebook principal para tener todo el
código en un único lugar, en vez de separarlo en un `ingest.py`.