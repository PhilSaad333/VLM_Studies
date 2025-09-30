# Bounding box tool

- First we just make the basic tool, with the ability to return a cropped version of the image as well

- Then test variations on how to have the model use based on just getting a sense of what helps the base model use it better

- Make a dataset for fine-tuning (I bet I can find one online)

- Make a dataset + reward for RL. Maybe the simplest thing would be to just use a smarter model as a judge of correctness, though I'd like to think of better options

# Experimentation with what affects the model's performance on fine-grained visual reasoning

- A first approach would be to identify problems (subset of NVLR) that count as visual reasoning. Or perhaps to keep things even simpler, take a subset of NVLR at which the model has low or modest pass rate

- Make new versions of those images (cropped, rotated, erase parts of it, etc) and see how performance changes with how the images are presented

- If theres a noticable difference, try to get a sense of what aspects of the ViT might be responsible. Maybe some simpler experiment would involve using a simpler ViT and seeing how these image variations affect the embeddings produced. For a CLIP trained ViT we could measure the inner product of the ViT embedding for these images and various text sequences describing them.