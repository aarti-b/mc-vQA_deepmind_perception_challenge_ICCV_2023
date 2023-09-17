# @title Prerequisites

#https://blog.paperspace.com/introduction-to-visual-question-answering/

import collections;
import json
import os
import random
from PIL import Image
from typing import List, Dict, Any, Tuple
import zipfile
import timm
import cv2
import csv
import math
import numpy as np
import requests
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from transformers import ViTFeatureExtractor, BertTokenizer, ViTModel, BertModel
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from transformers import ViTFeatureExtractor, BertTokenizer, ViTModel, BertModel
from torch import nn
from PIL import Image
from transformers import  PerceiverForImageClassificationLearned, AutoImageProcessor
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

def load_db_json(db_file: str) -> Dict[str, Any]:
    """Loads a JSON file as a dictionary.

  Args:
    db_file (str): Path to the JSON file.

  Returns:
    Dict: Loaded JSON data as a dictionary.

  Raises:
    FileNotFoundError: If the specified file doesn't exist.
    TypeError: If the JSON file is not formatted as a dictionary.
  """
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f'No such file: {db_file}')

    with open(db_file, 'r') as f:
        db_file_dict = json.load(f)
        if not isinstance(db_file_dict, dict):
            raise TypeError('JSON file is not formatted as a dictionary.')
        return db_file_dict


def load_mp4_to_frames(filename: str, target_height=224, target_width=224) -> np.array:
    """Loads an MP4 video file and returns its frames as a NumPy array.

  Args:
    filename (str): Path to the MP4 video file.
    target_height (int): Target height for resizing the frames.
    target_width (int): Target width for resizing the frames.

  Returns:
    np.array: Resized frames of the video as a NumPy array.
  """
    assert os.path.exists(filename), f'File {filename} does not exist.'
    print('THE FILENAME IS:', filename)
    cap = cv2.VideoCapture(filename)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vid_frames = np.empty((num_frames, target_height, target_width, 3), dtype=np.uint8)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (target_width, target_height))
        vid_frames[idx] = resized_frame
        idx += 1

    cap.release()
    return vid_frames



def get_video_frames(data_item: Dict[str, Any],
                     video_folder_path: str) -> np.array:
    """Loads frames of a video specified by an item dictionary.

  Assumes format of annotations used in the Perception Test Dataset.

  Args:
    data_item (Dict): Item from dataset containing metadata.
    video_folder_path (str): Path to the directory containing videos.

  Returns:
    np.array: Frames of the video as a NumPy array.
  """
    video_file = os.path.join(video_folder_path,
                            data_item['metadata']['video_id']) + '.mp4'
    vid_frames = load_mp4_to_frames(video_file)
    # assert data_item['metadata']['num_frames'] == vid_frames.shape[0]
    return vid_frames


# valid_db_path = './data/mc_question_valid.json'
train_db_path = '/storage/all_train.json' #'./data/mc_question_train.json'

# # load dataset annotations to dictionary
# valid_db_dict = load_db_json(valid_db_path)
train_db_dict = load_db_json(train_db_path)


# @title Dataset Class
class PerceptionDataset():
    """Dataset class to store video items from dataset.

      Attributes:
        video_folder: Path to the folder containing the videos.
        task: Task type for annotations.
        split: Dataset split to load.
        pt_db_list: List containing annotations for dataset according to
          split and task availability.
      """

    def __init__(self, pt_db_dict: Dict[str, Any], video_folder: str,
               task: str, split: str) -> None:
        """Initializes the PerceptionDataset class.

        Args:
          pt_db_dict (Dict): Dictionary containing annotations for dataset.
          video_folder (str): Path to the folder containing the videos.
          task (str): Task type for annotations.
          split (str): Dataset split to load.
        """
        self.video_folder = video_folder
        self.task = task
        self.split = split
        self.pt_db_list = self.load_dataset(pt_db_dict)

    def load_dataset(self, pt_db_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Loads the dataset from the annotation file and processes.

        Dict is processed according to split and task.

        Args:
          pt_db_dict: (Dict): Dictionary containing
            annotations.

        Returns:
          List: List of database items containing annotations.
        """
        pt_db_list = []
        for _, v in pt_db_dict.items():
            if v['metadata']['split'] == self.split:
                if v[self.task]:  # If video has annotations for this task
                    pt_db_list.append(v)

        return pt_db_list

    def __len__(self) -> int:
        """Returns the total number of videos in the dataset.

        Returns:
          int: Total number of videos.
        """
        return len(self.pt_db_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns the video and annotations for a given index.

          example_annotation = {
            'video_10909':{
              'mc_question': [
                {'id': 0, 'question': 'Is the camera moving or static?',
                      'options': ["I don't know", 'moving', 'static or shaking'],
                      'answer_id': 2, 'area': 'physics', 'reasoning': 'descriptive',
                      'tag': ['motion']
                }
              ]
            }
          }

        Args:
          idx (int): Index of the video.

        Returns:
          Dict: Dictionary containing the video frames, metadata, annotations.
        """
        data_item = self.pt_db_list[idx]
        annot = data_item[self.task]
        metadata = data_item['metadata']
        # here we are loading a placeholder as the frames
        # the commented out function below will actually load frames
        # vid_frames = np.zeros((metadata['num_frames'], 1, 1, 1))
        frames = get_video_frames(data_item, self.video_folder)

        return {'metadata': metadata,
                self.task: annot,
                'frames': frames}
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg_train = {'video_folder': '/storage/videos/',
       'task': 'mc_question',
       'split': 'train'}

train_dataset = PerceptionDataset(train_db_dict, **cfg_train)

import torch

import gc
gc.collect()
torch.cuda.empty_cache()
print(torch.__version__)


class VideoQuestionAnswering(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.resnext = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=0).to(device)
        self.config = resolve_data_config({}, model=self.resnext)
        self.transform = create_transform(**self.config)

        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=1, batch_first=True).to(device)

        self.miniLM = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(384 , 256) 
        self.fc2 = nn.Linear(512 , 384) 
        self.bn1 = nn.InstanceNorm1d(256)
        self.classifier = nn.Linear(256, num_classes) 


        # Weight Matrix - Vision modality 
        U = torch.Tensor(384, 384)
        self.U = nn.Parameter(U)

        # Weight Matrix - Text modality 
        V = torch.Tensor(384, 384)
        self.V = nn.Parameter(V)

        #Final Weight Matrix 
        W = torch.Tensor(384, 384)
        self.W = nn.Parameter(W)

        self.relu_f = nn.ReLU()
        self.sigmoid_f = nn.Sigmoid()
        
        # initialize weight matrices
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def process_frames(self, frames):
        # Process frames using ResNext and return the features
        features = []
        for frame in frames:
            frame_numpy = frame.cpu().detach().numpy()
            frame_image = Image.fromarray(frame_numpy.astype('uint8'))
            tensor = self.transform(frame_image).unsqueeze(0).to(device)
            out = self.resnext(tensor)
            features.append(out)

        features = torch.cat(features, dim=0)
        return features

    def forward(self, batch_frames, batch_questions, batch_options):
    
        all_video_logits = []

        
        # Iterate through each video in the batch
        for frames, questions, options in zip(batch_frames, batch_questions, batch_options):
            all_batch_logits = []
            video_logits_list = []
            num_frames = frames.shape[0]

            # Calculate the number of chunks for the current video
            num_chunks = (num_frames + 49) // 50


            for i in range(num_chunks):
                
                    # Get the start and end index for slicing
                    start_idx = i * 50
                    end_idx = min((i + 1) * 50, num_frames)

                    chunk_frames = frames[start_idx:end_idx]

                    # Process chunk of frames using ResNext
                    chunk_features = self.process_frames(chunk_frames)

                    # Pass the chunk features through LSTM
                    lstm_output, _ = self.lstm(chunk_features)

                    embedding_vector = lstm_output.mean(dim=0)  

                    all_batch_logits.append(embedding_vector.detach().cpu())
                    
                

            stacked_frames = torch.stack(all_batch_logits, dim=0)
            average_stacked_frames = stacked_frames.mean(dim=0).unsqueeze(0)
            frame_feature = self.fc2(average_stacked_frames.to(device))
            
            questions_list = []
            # Iterate through multiple-choice questions for each video
            # for question in questions:
            
            embeddings_LM = self.miniLM.encode(questions, convert_to_tensor=True)
            

            # Iterate through options and generate logits
            options_means_list = []

            for option in options:
                embedding_LM_options = self.miniLM.encode(option, convert_to_tensor=True)
                option_mean = embedding_LM_options.mean(dim=0)
                options_means_list.append(option_mean)

            options_tensor = torch.stack(options_means_list, dim=0)

            logits = []

            # Expand frame_feature from (1, 384) to (N, 384)
            expanded_frame_feature = frame_feature.expand(embeddings_LM.shape[0], -1).to(device)
            

            x1 = torch.nn.functional.normalize(expanded_frame_feature, p=2, dim=1)
            X_visual = torch.mm(x1, self.U.t())

            x_text = torch.nn.functional.normalize(embeddings_LM, p=2, dim=1)
            x_text2 = torch.nn.functional.normalize(options_tensor, p=2, dim=1)

            X_text = x_text + x_text2
            
            X_text = torch.mm(X_text, self.V.t())
            
            Xvt = X_visual * self.sigmoid_f(X_text)
            Xvt = self.relu_f(torch.mm(Xvt.view(-1, X_visual.size(-1)), self.W.t()))

            Xvt = self.fc1(Xvt)
            Xvt = self.bn1(Xvt)
            Xvt = self.dropout1(Xvt)

            logits = self.classifier(Xvt)

            logits_tensor = logits
            video_logits_list.append(logits_tensor)
            all_video_logits.append(video_logits_list)
            
        flattened_tensors = [tensor for sublist in all_video_logits for tensor in sublist]

        # Ensure all tensors have the same dimensionality before concatenation
        if not all([tensor.dim() == flattened_tensors[0].dim() for tensor in flattened_tensors]):
            raise ValueError("All tensors in flattened_tensors must have the same number of dimensions")

        # Concatenate tensors along dimension 0
        all_logits_tensor = torch.cat(flattened_tensors, dim=0)

        return all_video_logits, all_logits_tensor


    
class VideoQuestionAnsweringV2(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.resnext = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=0).to(device)
        self.config = resolve_data_config({}, model=self.resnext)
        self.transform = create_transform(**self.config)

        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=1, batch_first=True).to(device)

        self.miniLM = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(384 , 256) 
        self.fc2 = nn.Linear(512 , 384) 
        self.bn1 = nn.InstanceNorm1d(256)
        self.classifier = nn.Linear(256, num_classes) 


        # Weight Matrix - Vision modality 
        U = torch.Tensor(384, 384)
        self.U = nn.Parameter(U)

        # Weight Matrix - Text modality 
        V = torch.Tensor(384, 384)
        self.V = nn.Parameter(V)

        #Final Weight Matrix 
        W = torch.Tensor(384, 384)
        self.W = nn.Parameter(W)

        self.relu_f = nn.ReLU()
        self.sigmoid_f = nn.Sigmoid()
        
        # initialize weight matrices
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def process_frames(self, frames):
        # Process frames using ResNext and return the features
        features = []
        for frame in frames:
            frame_numpy = frame.cpu().detach().numpy()
            frame_image = Image.fromarray(frame_numpy.astype('uint8'))
            tensor = self.transform(frame_image).unsqueeze(0).to(device)
            out = self.resnext(tensor)
            features.append(out)

        features = torch.cat(features, dim=0)
        return features

    def forward(self, batch_frames, batch_questions, batch_options):

        all_video_logits = []
        
        # Iterate through each video in the batch
        for frames, questions, options in zip(batch_frames, batch_questions, batch_options):
            all_batch_logits = []
            video_logits_list = []


            # Define constants
            FRAME_LIMIT = 100
            CHUNK_SIZE = 50

            # Check if the video has less than 100 frames
            num_frames_to_process = min(frames.shape[0], FRAME_LIMIT)

            # Calculate the number of chunks for the frames to process
            num_chunks = (num_frames_to_process + CHUNK_SIZE - 1) // CHUNK_SIZE

            for i in range(num_chunks):
                # Get the start and end index for slicing
                start_idx = i * CHUNK_SIZE
                end_idx = start_idx + CHUNK_SIZE
                
                # Slice the frames
                chunk_frames = frames[start_idx:end_idx]
                
                # Process chunk of frames using ResNext
                chunk_features = self.process_frames(chunk_frames)

                # Pass the chunk features through LSTM
                lstm_output, _ = self.lstm(chunk_features)

                embedding_vector = lstm_output.mean(dim=0)  

                all_batch_logits.append(embedding_vector.detach().cpu())


            stacked_frames = torch.stack(all_batch_logits, dim=0)
            average_stacked_frames = stacked_frames.mean(dim=0).unsqueeze(0)
            frame_feature = self.fc2(average_stacked_frames.to(device))

            
            # Iterate through multiple-choice questions for each video
            # for question in questions:
            
            embeddings_LM = self.miniLM.encode(questions, convert_to_tensor=True)
            

            # Iterate through options and generate logits
            options_means_list = []

            for option in options:
                embedding_LM_options = self.miniLM.encode(option, convert_to_tensor=True)
                option_mean = embedding_LM_options.mean(dim=0)
                options_means_list.append(option_mean)

            options_tensor = torch.stack(options_means_list, dim=0)

            logits = []

            # Expand frame_feature from (1, 384) to (N, 384)
            expanded_frame_feature = frame_feature.expand(embeddings_LM.shape[0], -1).to(device)
            

            x1 = torch.nn.functional.normalize(expanded_frame_feature, p=2, dim=1)
            X_visual = torch.mm(x1, self.U.t())

            x_text = torch.nn.functional.normalize(embeddings_LM, p=2, dim=1)
            x_text2 = torch.nn.functional.normalize(options_tensor, p=2, dim=1)

            X_text = x_text + x_text2
            
            X_text = torch.mm(X_text, self.V.t())
            
            Xvt = X_visual * self.sigmoid_f(X_text)
            Xvt = self.relu_f(torch.mm(Xvt.view(-1, X_visual.size(-1)), self.W.t()))

            Xvt = self.fc1(Xvt)
            Xvt = self.bn1(Xvt)
            Xvt = self.dropout1(Xvt)

            logits = self.classifier(Xvt)

            logits_tensor = logits
            video_logits_list.append(logits_tensor)
            all_video_logits.append(video_logits_list)


        flattened_tensors = [tensor for sublist in all_video_logits for tensor in sublist]

        # Ensure all tensors have the same dimensionality before concatenation
        if not all([tensor.dim() == flattened_tensors[0].dim() for tensor in flattened_tensors]):
            raise ValueError("All tensors in flattened_tensors must have the same number of dimensions")

        # Concatenate tensors along dimension 0
        all_logits_tensor = torch.cat(flattened_tensors, dim=0)

        return all_video_logits, all_logits_tensor



def collate_batch(data):
    frames_list = [item['frames'] for item in data]
    actual_num_frames_list = [frames.shape[0] for frames in frames_list]

    max_frames = max(actual_num_frames_list)
    max_height = max(frames.shape[1] for frames in frames_list)
    max_width = max(frames.shape[2] for frames in frames_list)
    channels = frames_list[0].shape[3]

    batch_size = len(data)
    collated_frames = torch.zeros((batch_size, max_frames, max_height, max_width, channels), dtype=torch.uint8)

    questions_batch = []
    options_batch = []
    answer_ids_batch = []

    for i, (frames, actual_num_frames) in enumerate(zip(frames_list, actual_num_frames_list)):
        # Paddings for height and width
        pad_height = max_height - frames.shape[1]
        pad_width = max_width - frames.shape[2]

        padded_frames = F.pad(torch.tensor(frames[:actual_num_frames]), (0, 0, 0, pad_width, 0, pad_height))

        collated_frames[i, :actual_num_frames] = padded_frames

        questions = [q['question'] for q in data[i]['mc_question']]
        options = [q['options'] for q in data[i]['mc_question']]
        answer_ids = [q['answer_id'] for q in data[i]['mc_question']]

        questions_batch.append(questions)
        options_batch.append(options)
        answer_ids_batch.append(answer_ids)

    return {
        'frames': collated_frames,
        'questions': questions_batch,
        'options': options_batch,
        'answer_ids': answer_ids_batch
    }

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)


model = VideoQuestionAnsweringV2().to(device)
optimizer = Adam(model.parameters(), lr=0.001) # Learning rate
loss_function = CrossEntropyLoss()
accuracy = Accuracy(task="multiclass", num_classes=3, top_k=1).to(device)


# Set up a CSV writer
csv_file = open('training_log_resnext_half_gate.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Epoch', 'Average Loss', 'Top-1 Accuracy'])  # Write the header

# not using epochs for computation
epochs = 10
for epoch in range(epochs):

    loss_ep = 0
    init_batch=0
    pred_list = torch.tensor([], dtype=torch.long).to(device)
    answer_list = torch.tensor([], dtype=torch.long).to(device)

    num_batches = len(train_dataloader)
    print('Number of batches are {}'.format(num_batches))

    for dataset in train_dataloader:
        loss_cumulative = 0.0
        batch_frames = dataset['frames']
        questions = dataset['questions']
        options = dataset['options']
        answer_ids = dataset['answer_ids']
        answers = [item for sublist in answer_ids for item in sublist]
        answer = torch.tensor(answers).to(dtype=torch.long).to(device)

        print('------------------------------------------------------------')

        logits, flat_logits = model(batch_frames, questions, options)

        for i in range(len(logits)):
            batch_logits = logits[i][0]
            batch_answer_ids = answer_ids[i]

            loss = loss_function(batch_logits.to(device), torch.tensor(batch_answer_ids, dtype=torch.long).to(device))
            loss_cumulative += loss


        loss_cumulative.backward() # Compute gradients
        optimizer.step() # Update the parameters
        optimizer.zero_grad() # Clear gradients for the next step
        loss_ep += loss_cumulative.item()

        probabilities = F.softmax(flat_logits, dim=-1)
        highest_prob_index = torch.argmax(probabilities, dim=-1)

        print("Cross Entropy batch loss : {:.4f}".format(loss_cumulative))
        print('Batchwise top-1 accuracy is : {:.4f}'.format(accuracy(highest_prob_index, answer)))

        pred_list = torch.cat((pred_list, highest_prob_index))
        answer_list = torch.cat((answer_list, answer))
        top_1=accuracy(pred_list, answer_list)
        print('top-1 accuracy for epoch {} is : {:.4f}'.format(epoch, top_1))

        # Save epoch, avg_loss, and top-1 accuracy to CSV
        csv_writer.writerow([epoch, round(loss_cumulative.item(), 4), round(top_1.item(), 4)])
        csv_file.flush()
        init_batch+=1
        
        if init_batch%50==0:
            model_save_path = f'model_vqa_gated_resnext_half_epoch_{epoch}_{init_batch}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path} for epoch {epoch}')

    avg_loss_ep = loss_ep / num_batches
    top_1_ep=accuracy(pred_list, answer_list)
    print('top-1 accuracy batchwise for epoch {} is : {:.4f}'.format(epoch, top_1_ep))
    print('Cross Entropy Loss for epoch {} : {:.4f}'.format(epoch, avg_loss_ep))
    csv_writer.writerow([epoch, round(avg_loss_ep, 4), round(top_1_ep.item(), 4)])
    csv_file.flush()

    # save model after each epochs
    model_save_path = f'model_vqa_gated_resnext_half_epoch_{epoch}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path} for epoch {epoch}')

model_save_path = 'final_model_VQA_gated_resnext_half.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

csv_file.close()
