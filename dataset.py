import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, visit_info,  label):
        self.x_lines = visit_info
        self.y_lines = label

    def __len__(self):
        return len(self.x_lines)

    def __getitem__(self, index):
        visit_diag_code = self.x_lines[index]
        visit_label = self.y_lines[index]
        return visit_diag_code, visit_label

    @staticmethod
    def collate_fn(batch):
        # This setting is for heart failure 
        maxVisitTime = 460
        maxCodeLengh = 50 
        padding_idx = 8692


        x_result = []
        padding_result = []  
        label_result = [] 
        y_result = []    

        for b, l in batch:
            x_result.append(b)
            y_result.append(l)

            j_list = []
            for j in b:
                code_padding_amount = maxCodeLengh - len(j)
                j_list.append(j + [padding_idx] * code_padding_amount)
            padding_result.append(j_list)

            visit_padding_amount = maxVisitTime - len(b)
            for time in range(visit_padding_amount):
                padding_result[-1].append([padding_idx] * maxCodeLengh)
        
        return (torch.tensor(padding_result), torch.tensor(y_result), x_result, y_result)
