import sys
sys.stdout.reconfigure(encoding='utf-8')

import gradio as gr
from gradio_multisteptasklist import MultiStepTaskList
from gradio_client import Client, handle_file



# example = MultiStepTaskList().example_value()

# demo = gr.Interface(
#     lambda x:x,
#     MultiStepTaskList(),  # interactive version of your component
#     MultiStepTaskList(),  # static version of your component
#     # examples=[[example]],  # uncomment this line to view the "example version" of your component
# )


# if __name__ == "__main__":
#     demo.launch()
def add_thought(thoughts, new_thought):
    if thoughts is None:
        thoughts = []
    if new_thought and new_thought.strip():
        thoughts.append(new_thought.strip())
    return thoughts, "", thoughts 

# with gr.Blocks() as demo:
#     state = gr.State([])
#     textbox = gr.Textbox(label="Add a new agent thought", 
#     placeholder="Type a thought and press Enter…")
#     viewer = MultiStepTaskList(label="Agent Thoughts")
#     textbox.submit(add_thought, [state, textbox], [viewer, textbox, state])
def call_my_api(file, question):
    client = Client("http://127.0.0.1:7860")  # Adjust the URL as needed
    result = client.predict(
        handle_file(file.name),  # file.name is the uploaded temp file path
        question,
        api_name="/predict"
    )
    print("RESULT FROM API:", result)
    answer = result.get("answer", "")
    thoughts = result.get("thoughts", [])
    # Optionally, handle references here
    return answer, thoughts  # Return both

iface = gr.Interface(
    fn=call_my_api,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Ask a question"),
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        MultiStepTaskList(label="Agent Thoughts")  # Custom component for thoughts!
    ],
    title="PDF Q&A with Agent Thoughts",
    description="Upload a PDF and ask a question. See the answer and the agent's reasoning steps."
)

iface.launch()
