
import gradio as gr
from gradio_multisteptasklist import MultiStepTaskList


example = MultiStepTaskList().example_value()

demo = gr.Interface(
    lambda x:x,
    MultiStepTaskList(),  # interactive version of your component
    MultiStepTaskList(),  # static version of your component
    # examples=[[example]],  # uncomment this line to view the "example version" of your component
)


if __name__ == "__main__":
    demo.launch()
