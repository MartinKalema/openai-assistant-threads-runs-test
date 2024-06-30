from openai import OpenAI
import os 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# create assistant
assistant = client.beta.assistants.create(
    instructions="You are a personal data science assistant. Use your knowledge base to answer questions about tokenization in Large Language Models",
    name="Opter",
    tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
    model="gpt-4-1106-preview",
    temperature=0.6
)

# create a vector store to house your files
vector_store = client.beta.vector_stores.create(name="Research Papers")

file_paths = ["2305.15425v2.pdf"]
file_streams = [open(path, "rb") for path in file_paths]

file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)

print(file_batch.status)
print(file_batch.file_counts)

# Update assistant to use new vector store.
assistant = client.beta.assistants.update(
  assistant_id=assistant.id,
  tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

# You can also attach files as message attachments to your thread. Doing so will create a vector store associated with the thread. If there is already a vector store associated with this thread, the new file will be attached to the existing vector store.

# When you run this thread, the file_search tool will Query both the vector store from your assistant and also the vector store on the thread.

message_file = client.files.create(
  file=open("2010.11934v3.pdf", "rb"), purpose="assistants"
)

# Create a thread
thread = client.beta.threads.create(
  messages=[
    {
      "role": "user",
      "content": "How does tokenization affect model performance and how is it done in the google/mt5 model?",
      "attachments": [
        { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
      ],
    }
  ]
)
 
print(thread.tool_resources.file_search)

# Create a run
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id, assistant_id=assistant.id
)

messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

# messages[0].content[0].text.value


message_content = messages[0].content[0].text
annotations = message_content.annotations
citations = []
for index, annotation in enumerate(annotations):
    message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
    if file_citation := getattr(annotation, "file_citation", None):
        cited_file = client.files.retrieve(file_citation.file_id)
        citations.append(f"[{index}] {cited_file.filename}")

print(message_content.value)
print("\n".join(citations))
