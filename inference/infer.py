# @title Prerequisites

#https://blog.paperspace.com/introduction-to-visual-question-answering/

import collections;
import json
import os
import random
from typing import List, Dict, Any, Tuple
import zipfile
import gc
import timm
import cv2
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
# import moviepy.editor as mvp
import numpy as np
import requests
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from transformers import ViTFeatureExtractor, BertTokenizer, ViTModel, BertModel
from torch.nn.utils.rnn import pad_sequence

from transformers import ViTFeatureExtractor, BertTokenizer, ViTModel, BertModel


from PIL import Image
# from transformers import VivitImageProcessor, VivitModel
from transformers import AutoImageProcessor, TimesformerModel, TimesformerConfig

import math
from sentence_transformers import SentenceTransformer

from torch import nn
from PIL import Image
from PIL import Image
from transformers import VivitImageProcessor, VivitModel
import math
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # print('THE FILENAME IS:', filename)
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
test_db_path = '/storage/all_test.json'

# # load dataset annotations to dictionary
# valid_db_dict = load_db_json(valid_db_path)
test_db_dict = load_db_json(test_db_path)


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

    frames = get_video_frames(data_item, self.video_folder)

    return {'metadata': metadata,
            self.task: annot,
            'frames': frames,
            }

cfg_test = {'video_folder': '/storage/videos/',
       'task': 'mc_question',
       'split': 'test'}

test_dataset = PerceptionDataset(test_db_dict, **cfg_test)



class VideoQuestionAnsweringVivit(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        '''
        vision model for video encoding
        '''
        self.feature_extractor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.vivit = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", ignore_mismatched_sizes=True).to(device)

        '''
        text encoding for texts
        '''
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)

        self.dropout1 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(768 , 384)
        # self.bn1 = nn.BatchNorm1d(384)
        self.bn1 = nn.InstanceNorm1d(384)

        self.classifier = nn.Linear(384, num_classes)

        # Weight Matrix - Vision modality
        U = torch.Tensor(768, 768)
        self.U = nn.Parameter(U)

        # Weight Matrix - Text modality
        V = torch.Tensor(768, 768)
        self.V = nn.Parameter(V)

        #Final Weight Matrix
        W = torch.Tensor(768, 768)
        self.W = nn.Parameter(W)

        self.relu_f = nn.ReLU()
        self.sigmoid_f = nn.Sigmoid()

        # initialize weight matrices
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))


        # self.fc = nn.Linear(768 + self.bert.config.hidden_size*2, num_classes)

    def forward(self, batch_frames, batch_questions, batch_options):
        all_batch_logits = []

        frame_features = torch.zeros(1, 768)

        for frames, questions, options in zip(batch_frames, batch_questions, batch_options):

            # Iterate through each video in the batch
            num_frames = frames.shape[0]  # Number of frames in the current video

            # Calculate the number of chunks for the current video
            num_chunks = (num_frames + 31) // 32

            for i in range(num_chunks):
                # Get the start and end index for slicing
                start_idx = i * 32
                end_idx = min((i + 1) * 32, num_frames)  # Ensure we don't go beyond the available frames for the last chunk

                chunk_frames = frames[start_idx:end_idx]

                # Create a tensor filled with zeros of the desired shape (always 32 frames)
                padded_chunk_frames = torch.zeros((32,) + chunk_frames.shape[1:], dtype=chunk_frames.dtype)

                # Copy the actual frames into this tensor
                padded_chunk_frames[:chunk_frames.shape[0]] = chunk_frames

                inputs = self.feature_extractor(list(padded_chunk_frames), return_tensors="pt")
                # print(inputs.pixel_values.shape)

                # forward pass
                outputs = self.vivit(**inputs.to(device))
                last_hidden_states = outputs.last_hidden_state
                embedding_vector = last_hidden_states.mean(dim=1)

                #mean of all frames
                frame_features = (frame_features * i + embedding_vector.cpu().detach()) / (i + 1)
                gc.collect()
                torch.cuda.empty_cache()


            all_logits = []
            questions_list = []
            # Iterate through multiple-choice questions for each video
            for question in questions:

                question_inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                question_outputs = self.bert(**question_inputs)
                question_features = question_outputs.pooler_output
                # print(question_features.shape)
                questions_list.append(question_features)


            # Iterate through options and generate logits

            options_list_global =[]
            for option in options:
                options_list = []
                for i in range(3):
                    option_inputs = self.tokenizer(option[i], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                    option_outputs = self.bert(**option_inputs)
                    option_features = option_outputs.pooler_output
                    # print(option_features.shape)
                    options_list.append(option_features)
                options_tensor = torch.stack(options_list)

                # Calculate the mean along the batch dimension
                opt_features = options_tensor.mean(dim=0)
                # print(opt_features.shape)
                options_list_global.append(opt_features)

            logits = []

            for i in range(len(questions_list)):

                x1 = torch.nn.functional.normalize(frame_features.to(device), p=2, dim=1)
                X_visual = torch.mm(x1, self.U.t())

                x2 = questions_list[i]
                x_text = torch.nn.functional.normalize(x2, p=2, dim=1)

                x3 = options_list_global[i]
                x_text2 = torch.nn.functional.normalize(x3, p=2, dim=1)

                X_text = x_text + x_text2
                X_text = torch.mm(X_text, self.V.t())
                # print(X_visual.shape, x_text.shape, x_text2.shape, X_text.shape)

                Xvt = X_visual * self.sigmoid_f(X_text)
                Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))
                # print(Xvt.shape)

                Xvt = self.fc1(Xvt)
                Xvt = self.bn1(Xvt)
                Xvt = self.dropout1(Xvt)

                logit = self.classifier(Xvt)

                logits.append(logit.squeeze())


            all_logits.append(torch.stack(logits))

            all_batch_logits.append(all_logits)

            # print(all_batch_logits)
            flat_list = [tensor for sublist in all_batch_logits for tensor in sublist]

            # Concatenate the tensors along the first dimension
            logits_tensor = torch.cat(flat_list, dim=0)

        return all_batch_logits, logits_tensor

    
class VideoQuestionAnsweringTimesformer(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        

        '''
        vision model for video encoding 
        '''

        self.feature_extractor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        # self.times = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device)

        # Load the configuration of the pre-trained model
        config = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")

        # Modify the specific configuration attribute (assuming `num_frames` exists, which it might not)
        config.num_frames = 32

        # Initialize the model with this modified configuration
        self.times = TimesformerModel(config).to(device)
        
        '''
        text encoding for texts 
        '''
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)

        self.dropout1 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(768 , 384) 
        # self.bn1 = nn.BatchNorm1d(384)
        self.bn1 = nn.InstanceNorm1d(384)

        self.classifier = nn.Linear(384, num_classes) 

        # Weight Matrix - Vision modality 
        U = torch.Tensor(768, 768)
        self.U = nn.Parameter(U)

        # Weight Matrix - Text modality 
        V = torch.Tensor(768, 768)
        self.V = nn.Parameter(V)

        #Final Weight Matrix 
        W = torch.Tensor(768, 768)
        self.W = nn.Parameter(W)

        self.relu_f = nn.ReLU()
        self.sigmoid_f = nn.Sigmoid()
        
        self.fusion_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        
        # initialize weight matrices
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        

        # self.fc = nn.Linear(768 + self.bert.config.hidden_size*2, num_classes)

    def forward(self, batch_frames, batch_questions, batch_options):
        all_batch_logits = []
        
        frame_features = torch.zeros(1, 768)
        
        for frames, questions, options in zip(batch_frames, batch_questions, batch_options):

            
            # Iterate through each video in the batch
            num_frames = frames.shape[0]  # Number of frames in the current video

            # Calculate the number of chunks for the current video
            num_chunks = (num_frames + 31) // 32
            
            for i in range(num_chunks):
                # Get the start and end index for slicing
                start_idx = i * 32
                end_idx = min((i + 1) * 32, num_frames)  # Ensure we don't go beyond the available frames for the last chunk
                
                chunk_frames = frames[start_idx:end_idx]

                # Create a tensor filled with zeros of the desired shape (always 32 frames)
                padded_chunk_frames = torch.zeros((32,) + chunk_frames.shape[1:], dtype=chunk_frames.dtype)

                # Copy the actual frames into this tensor
                padded_chunk_frames[:chunk_frames.shape[0]] = chunk_frames
                
                inputs = self.feature_extractor(list(padded_chunk_frames), return_tensors="pt")
                # print(inputs.pixel_values.shape)

                # forward pass
                outputs = self.times(**inputs.to(device))
                last_hidden_states = outputs.last_hidden_state
                embedding_vector = last_hidden_states.mean(dim=1)

                #mean of all frames 
                frame_features = (frame_features * i + embedding_vector.cpu().detach()) / (i + 1)
                gc.collect()
                torch.cuda.empty_cache()


            all_logits = []
            questions_list = []
            # Iterate through multiple-choice questions for each video
            for question in questions:

                question_inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                question_outputs = self.bert(**question_inputs)
                question_features = question_outputs.pooler_output
                
                questions_list.append(question_features)


            # Iterate through options and generate logits

            options_list_global =[]
            for option in options:
                options_list = []
                for i in range(3):
                    option_inputs = self.tokenizer(option[i], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                    option_outputs = self.bert(**option_inputs)
                    option_features = option_outputs.pooler_output
                    # print(option_features.shape)
                    options_list.append(option_features)
                options_tensor = torch.stack(options_list)

                # Calculate the mean along the batch dimension
                opt_features = options_tensor.mean(dim=0)
                # print(opt_features.shape)
                options_list_global.append(opt_features)

            logits = []

            for i in range(len(questions_list)):
  
                x1 = torch.nn.functional.normalize(frame_features.to(device), p=2, dim=1)
                X_visual = torch.mm(x1, self.U.t())

                x2 = questions_list[i]
                x_text = torch.nn.functional.normalize(x2, p=2, dim=1)

                x3 = options_list_global[i]
                x_text2 = torch.nn.functional.normalize(x3, p=2, dim=1)

                X_text = x_text + x_text2
                X_text = torch.mm(X_text, self.V.t())
                    
                Xvt = X_visual * self.sigmoid_f(X_text)

                # Self-Attention on the fused features
                attention_output, _ = self.fusion_attention(Xvt.unsqueeze(0), Xvt.unsqueeze(0), Xvt.unsqueeze(0))
                Xvt = attention_output.squeeze(0)

                Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))
                
                Xvt = self.fc1(Xvt)
                Xvt = self.bn1(Xvt)
                Xvt = self.dropout1(Xvt)


                logit = self.classifier(Xvt)
                
                logits.append(logit.squeeze())


            all_logits.append(torch.stack(logits))

            all_batch_logits.append(all_logits)

            # print(all_batch_logits)
            flat_list = [tensor for sublist in all_batch_logits for tensor in sublist]

            # Concatenate the tensors along the first dimension
            logits_tensor = torch.cat(flat_list, dim=0)

        return all_batch_logits, logits_tensor 


class VideoQuestionAnsweringPerceiver(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.feature_extractor = AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-learned")
        self.vit = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned").to(device)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)


        self.dropout1 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1024 , 768) 
        self.bn1 = nn.InstanceNorm1d(768)

        self.classifier = nn.Linear(768, num_classes) 

        # Weight Matrix - Vision modality 
        U = torch.Tensor(1024, 1024)
        self.U = nn.Parameter(U)

        # Weight Matrix - Text modality 
        V = torch.Tensor(768, 768)
        self.V = nn.Parameter(V)

        #Final Weight Matrix 
        W = torch.Tensor(1024, 1024)
        self.W = nn.Parameter(W)

        self.relu_f = nn.ReLU()
        self.sigmoid_f = nn.Sigmoid()
        
        # initialize weight matrices
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, batch_frames, batch_questions, batch_options):
        all_batch_logits = []
        frame_features = torch.zeros(1, 1024)
        
        # Iterate through each video in the batch
        for frames, questions, options in zip(batch_frames, batch_questions, batch_options):
            count=0
            for i, framei in enumerate(frames):
                
                frame_numpy = framei.cpu().detach().numpy()

                frame = Image.fromarray(frame_numpy.astype('uint8'))
                inputs = self.feature_extractor(images=frame, return_tensors="pt").pixel_values

                outputs = self.vit(inputs=inputs.to(device), output_attentions=False, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1]
                embedding_vector = embeddings.mean(dim=1)

                count+=1
                # Compute running mean
                frame_features = (frame_features * i + embedding_vector.cpu().detach()) / (i + 1)
                gc.collect()
                torch.cuda.empty_cache()


            all_logits = []
            questions_list = []
            # Iterate through multiple-choice questions for each video
            for question in questions:

                question_inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                question_outputs = self.bert(**question_inputs)
                question_features = question_outputs.pooler_output
                
                questions_list.append(question_features)


            # Iterate through options and generate logits

            options_list_global =[]
            for option in options:
                options_list = []
                for i in range(3):
                    option_inputs = self.tokenizer(option[i], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                    option_outputs = self.bert(**option_inputs)
                    option_features = option_outputs.pooler_output
                    options_list.append(option_features)
                options_tensor = torch.stack(options_list)

                # Calculate the mean along the batch dimension
                opt_features = options_tensor.mean(dim=0)
                # print(opt_features.shape)
                options_list_global.append(opt_features)

            logits = []

            for i in range(len(questions_list)):
                X_visual = torch.nn.functional.normalize(frame_features.to(device), p=2, dim=1)

                x2 = questions_list[i]
                x_text = torch.nn.functional.normalize(x2, p=2, dim=1)

                x3 = options_list_global[i]
                x_text2 = torch.nn.functional.normalize(x3, p=2, dim=1)

                # combined = torch.cat([frame_features, questions_list[i], options_list_global[i]], dim=-1)
                X_text = x_text + x_text2
                X_text = torch.mm(X_text, self.V.t())
                # print(X_visual.shape, x_text.shape, x_text2.shape, X_text.shape)
                padding_size = X_visual.size(1) - X_text.size(1)

                # Pad X_text to have the same size as X_visual along dimension 1
                X_text_padded = F.pad(X_text, (0, padding_size))

                # Now you can do the element-wise multiplication
                Xvt = X_visual * self.sigmoid_f(X_text_padded)
                                    
                Xvt = X_visual * self.sigmoid_f(X_text_padded)
                Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))
                # print(Xvt.shape)
                
                Xvt = self.fc1(Xvt)
                Xvt = self.bn1(Xvt)
                Xvt = self.dropout1(Xvt)

                logit = self.classifier(Xvt)

                logits.append(logit.squeeze())


            all_logits.append(torch.stack(logits))

            all_batch_logits.append(all_logits)

            # print(all_batch_logits)
            flat_list = [tensor for sublist in all_batch_logits for tensor in sublist]

            # Concatenate the tensors along the first dimension
            logits_tensor = torch.cat(flat_list, dim=0)

        return all_batch_logits, logits_tensor
    
    
class VideoQuestionAnsweringResnext(nn.Module):
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

    for i, (frames, actual_num_frames) in enumerate(zip(frames_list, actual_num_frames_list)):
        # Paddings for height and width
        pad_height = max_height - frames.shape[1]
        pad_width = max_width - frames.shape[2]

        padded_frames = F.pad(torch.tensor(frames[:actual_num_frames]), (0, 0, 0, pad_width, 0, pad_height))

        collated_frames[i, :actual_num_frames] = padded_frames

        questions = [q['question'] for q in data[i]['mc_question']]
        options = [q['options'] for q in data[i]['mc_question']]



        questions_batch.append(questions)
        options_batch.append(options)

    video_id = data[0]['metadata']['video_id']


    return {
        'frames': collated_frames,
        'questions': questions_batch,
        'options': options_batch,
        'video_id':video_id,
    }

from torch.utils.data import DataLoader, Dataset, SequentialSampler
indices = list(range(len(test_dataset) - 1, -1, -1))
reverse_sampler = SequentialSampler(indices)


test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, sampler=reverse_sampler, collate_fn=collate_batch)

model = VideoQuestionAnsweringResnext().to(device)
state_dict = torch.load('/notebooks/model_vqa_gated_resnext_half_epoch_2.pth', map_location=torch.device('cuda'))
model.load_state_dict(state_dict)
model.eval()

import json
import os

# Load previous results if the file exists
previous_results = {}
if os.path.exists('/notebooks/test_results_half_resnext.json'):
    with open('/notebooks/test_results_half_resnext.json', 'r') as my_file:
        previous_results = json.load(my_file)

submission = {}
count = 0
for dataset in test_dataloader:
    video_id = dataset['video_id']

    # Skip the video id if it's already been processed
    if video_id in previous_results:
        count+=1
        print(f"Skipping video id: {video_id} as it's already processed with count {count}.")
        continue

    # Extract relevant data from the dataset
    batch_frames = dataset['frames']
    questions = dataset['questions']
    options = dataset['options']

    # Pass the data through the model
    logits, flat_logits = model(batch_frames, questions, options)
    probabilities = F.softmax(flat_logits, dim=-1)
    highest_prob_indices = torch.argmax(probabilities, dim=-1).cpu().detach().numpy()

    video_predictions = []
    for q_id, answer_index in enumerate(highest_prob_indices):
        video_predictions.append({
            'id': int(q_id),
            'answer_id': int(answer_index),
            'answer': options[0][q_id][answer_index]
        })

    submission[video_id] = video_predictions
    count += 1
    print('Count {} was done.'.format(count))

    # Save results in real-time
    with open('/notebooks/test_results_half_resnext.json', 'w') as my_file:
        json.dump({**previous_results, **submission}, my_file)



