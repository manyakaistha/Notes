import torch

torch.manual_seed(42)
truth=torch.tensor([1,2,3,4])
pred=torch.tensor([1,7,6,4])
# At each training step the next input is the gold previous token.
teacher_input=torch.cat([torch.tensor([0]),truth[:-1]])
# Free-running inference consumes the model's previous prediction.
free_input=torch.cat([torch.tensor([0]),pred[:-1]])
print(f"gold target={truth.tolist()}")
print(f"teacher-forced decoder input={teacher_input.tolist()}")
print(f"free-running decoder input={free_input.tolist()}")
print("teacher forcing exposes gold prefix; free-running errors can enter later inputs")
