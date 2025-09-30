# VLM Studies

Project to understand multimodal AI and build an experimentation "laboratory" around visual-language models. My main goal is to get some familiarity with multimodal AI. For now, I'm just sticking with models that can take images as input, postponing anything about audio and video, as well as generating audio/images/video to later.

In order to do this I'm focusing on:

- Doing lots of eval on multimodal AI models (just Qwen2-VL for now) to get a sense of their strengths and weaknesses
- Coming up with some "things to do" with a multimodal model that might be interesting, probably motivated by trying to see how to address a weakness of the model
- Finding questions to study in a scientific way that will help me understand why the model is behaving in a certain way and how to change that. 

My first 'mini-project' after spending a little bit of time testing out Qwen2-VL revolves around the model's ability to precisely keep track of multiple objects in an image. For example, one task it failed badly at involved an image with several multiple TV screens. It failed to count the screens, and failed even more badly at listing what was on the screens.

This failure motivated the following "things to do" and "questions to study":

For a "thing to do" - What sort of things could we do to improve the model's capability for these kinds of tasks? I thought it might be interesting to try out teaching the model to use a bounding box tool to create boxes on the image with accompanying labels. The model could further be trained to use this tool while reasoning about questions like this (e.g. it could go through each object in order and box+label them, then at the end it has a clear list).

Of course there are a lot of questions that this brings up. Some that stood out to me more are along the lines of "Do different ways of presenting the visual information have an effect on the model's ability to understand the image in a fine-grained way (e.g. to be able to correctly count and describe several similar objects)? How does the way the model pre-processes images (for Qwen2-VL, the ViT does patching and then we do resampling) affect its performance in these sorts of tasks? As a basic question that I'm sure has been studied to death already, is performance on tasks like this sensitive to the details of the patching and re-sampling? Does reorienting the image make a difference (perhaps how many patches an object is in initially makes a difference?)? Does this motivate changes to my bounding box tool approach (Should the bounding box tool create a cropped version of the image in addition to one marked up with a bounding box?) 

Update -

Today I read about two relevant things for this project

- Huggingface recently released their "Smol2Operator" https://huggingface.co/blog/smol2operator, a 2b model trained to navigate computer interfaces. Their success with a small model like this is encouraging, and their datasets are probably going to be very useful.

- John Schulman at Thinking Machines wrote this blog https://thinkingmachines.ai/blog/lora/ about the efficacy of LoRA. One of the many takeaways I had was about estimating the information to be gained from a dataset vs the capacity of the model. Previously I had just blindly chosen LoRA parameters. However I'd like to be able to do better in this project to be extra efficient. Apparently we should roughly think of each trainable paramter as having the capacity to learn 2 bits of information (My impression from briefly skimming the paper he cited on this is that this is basically just an emperical observation, not clear why this value and even why there is a sort of universal capacity). If we are interested in fine tuning on some dataset, it's a good idea then to measure the information to be gained from the dataset. Apparently for typical LLM datasets, it's typically about one bit per token. This can be measured by computing the mean surprisal per token of the to-be-finetuned model on the fine-tuning dataset. 

So it seems worthwhile to get a sense of the excess information (as in the amount of information to be gained by the to-be-finetuned model) contained in the image datasets I'll be using (as measured by the surprisal). I'd want to choose LoRA $k$ such that the number of lora parameters is well above this.

It would also be interesting to explore what factors affect how much information is to be gained from images (plenty to do here I think)