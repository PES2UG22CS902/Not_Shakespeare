import os
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16
).to("cuda")


@tool("Image Generator")
def generate_image(prompt: str) -> str:
    """Generates an image from a text prompt using Stable Diffusion."""
    image = pipe(prompt, num_inference_steps=25).images[0]
    image_path = f"generated_images/{prompt.replace(' ', '_')[:50]}.png"
    os.makedirs("generated_images", exist_ok=True)
    image.save(image_path)
    return image_path


@tool("Image Captioning")
def caption_image(image_path: str) -> str:
    """Generates a caption for an image using BLIP."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


@tool("Markdown Compiler")
def create_markdown(story: str, image_paths: list, captions: list) -> str:
    """Compiles story, images, and captions into a Markdown file."""
    markdown_content = "--Peak Cinema--\n\n"

    paths = image_paths.split(",") if isinstance(image_paths, str) else image_paths
    cap_list = captions.split(",") if isinstance(captions, str) else captions

    for i, img in enumerate(paths):
        img = img.strip()
        caption = cap_list[i].strip() if i < len(cap_list) else "No caption available"
        markdown_content += f"![Image]({img})\n\n*{caption}*\n\n"

    markdown_content += f"## Story\n\n{story}"

    with open("story.md", "w") as f:
        f.write(markdown_content)

    return "Markdown file generated successfully at story.md"


llm_config = LLM(model="ollama/gemma3:4b", base_url="http://localhost:11434")

story_agent = Agent(
    role="Storyteller Agent",
    goal="Give a short insightful story",
    backstory="You love storytelling. You're an absolute powerhouse. You'll talk about stories that have deeper morals about humanity.",
    llm=llm_config
)

image_agent = Agent(
    role="Image Generator Agent",
    goal="Create compelling images for the story",
    llm=llm_config,
    backstory="You love image generation. You've closely studied cinematography like that of Roger Deakins. You are an AI artist who brings stories to life with visually stunning images.",
    tools=[generate_image]
)

caption_agent = Agent(
    role="Captioning Agent",
    goal="Provide meaningful captions for the images",
    llm=llm_config,
    backstory="You are an expert in describing images in an insightful and concise manner.",
    tools=[caption_image]
)

markdown_agent = Agent(
    role="Markdown Compiler Agent",
    goal="Assemble the final Markdown file with story, images, and captions",
    backstory="You are a meticulous documenter, formatting AI-generated content into structured Markdown files.",
    llm=llm_config,
    tools=[create_markdown]
)

prompt = input("Gimme a prompt and I'll give you absolute cinema >:) -- ")

story_task = Task(
    description = f"Write a story about, {prompt}, in a neat format.",
    expected_output = "Give me a story.",
    agent=story_agent
)

image_task = Task(
    description="Generate images for key scenes in the story. Use the story's theme and scenes to create compelling visuals. Return a list of image file paths.",
    expected_output="Paths to generated images",
    agent=image_agent,
    context=[story_task]
)

caption_task = Task(
    description="Create meaningful captions for each of the generated images. Each caption should relate to the story and describe what's happening in the image.",
    expected_output="Captions for each image",
    agent=caption_agent,
    context=[image_task, story_task]
)

markdown_task = Task(
    description="Compile the story, images, and captions into a Markdown file. The file should have a title, images with captions, and then the story text.",
    expected_output="A completed Markdown file",
    agent=markdown_agent,
    context=[story_task, image_task, caption_task]
)

crew = Crew(
    agents=[story_agent, image_agent, caption_agent, markdown_agent],
    tasks=[story_task, image_task, caption_task, markdown_task],
    process=Process.sequential
)

result = crew.kickoff()

print("#########################################################################")
print("Read story.md for cinematic masterpiece.")
print(result)