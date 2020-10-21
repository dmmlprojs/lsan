import argparse
from torch.utils.data import DataLoader
from trainer import *
from transformer import *
from dataset import *
from LSAN import *
import pickle



parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_path", default='output/result.txt', help="Output Path")
parser.add_argument("-hs", "--hidden", type=int, default=256, help="Embedding size of Transformer")
parser.add_argument("-l", "--layers", type=int, default=8, help="Number of Layers of Transformer")
parser.add_argument("-a", "--attn_heads", type=int, default=8, help="Number of Transformer Heads")
parser.add_argument("-drop", "--dropout", type=float, default=0.1)

parser.add_argument("-b", "--batch_size", type=int, default=64, help="Number of Batch Size")
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of Epochs")
parser.add_argument("-w", "--num_workers", type=int, default=0, help="Dataloader Worker Size")

parser.add_argument("-gpu","--with_cuda", type=int, default=0, help="Training with GPU")

parser.add_argument("-lr","--lr", type=float, default=0.001, help="Learning Rate of Adam")
parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight Decay of Adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam First Beta Value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam First Beta Value")

parser.add_argument("--saving_path", default='model_parameters/models.pth', help="Path for Saving the Models")

args = parser.parse_args()

# Setting new dataset and dataloader
with open('model_inputs/training.pickle','rb') as f:
    train_data = pickle.load(f)
train_visit = train_data[0]
train_label = train_data[1]
train_dataset = Dataset(train_visit, train_label)

with open('model_inputs/validation.pickle','rb') as f:
    validate_data = pickle.load(f)
validate_visit = validate_data[0]
validate_label = validate_data[1]
validate_dataset = Dataset(validate_visit, validate_label)

with open('model_inputs/testing.pickle','rb') as f:
    test_data = pickle.load(f)
test_visit = test_data[0]
test_label = test_data[1]
test_dataset = Dataset(test_visit, test_label)



print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn,shuffle=False)
validate_data_loader = DataLoader(validate_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn,shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn,shuffle=False)




embedding_dim = args.hidden
code2idx_file = 'code2idx_new.pickle'



with open(code2idx_file, 'rb') as f:
    code2idx = pickle.load(f)
    #  list out keys and values separately
    diagnosis_code_list = list(code2idx.keys())
    dignosis_index_list = list(code2idx.values())

model = LSAN(len(dignosis_index_list), embedding_dim, transformer_hidden = args.hidden, attn_heads = args.attn_heads,
                    transformer_dropout = args.dropout, transformer_layers = args.layers)

# defining trainer
LSAN_train = LSAN_trainer(model, train_dataloader=train_data_loader, validate_dataloader=validate_data_loader, test_dataloader=test_data_loader, with_cuda=args.with_cuda, lr=args.lr, output_dir=args.output_path)

# training process
for epoch in range(args.epochs):
    LSAN_train.train(epoch)
 
    # Validation
    if validate_data_loader is not None:
        LSAN_train.validate(epoch)
      
torch.save(model.state_dict(), args.saving_path)

# Test after the end of training
test_model = LSAN(len(dignosis_index_list), embedding_dim, transformer_hidden = args.hidden, attn_heads = args.attn_heads, transformer_dropout = args.dropout, transformer_layers = args.layers) 

test_model.load_state_dict(torch.load(args.saving_path))
test_model.eval()

LSAN_test = LSAN_trainer(test_model, train_dataloader=train_data_loader, validate_dataloader=validate_data_loader, test_dataloader=test_data_loader, with_cuda=args.with_cuda, lr=args.lr, output_dir=args.output_path)

LSAN_test.test(epoch)
