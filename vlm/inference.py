import torch
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import pipeline
import glob
from numpy import genfromtxt
import numpy as np
import os
from pathlib import Path


def read_videos_and_step_start_end():
    step_csv_path = 'NIV/csvs'
    all_csvs = glob.glob(step_csv_path + '/*')
    all_csvs = sorted(all_csvs)
    
    video_names = []
    for i in range(len(all_csvs)):
        video_names.append(all_csvs[i].split('/')[-1].split('.')[0])
    #print(video_names)
    
    csv_path = []
    vid_path = []
    for i in range(len(video_names)):
        csv_path.append('NIV/csvs/' + video_names[i] + '.csv')
        vid_path.append('NIV/segmented_start_end/' + video_names[i])
                        
    return csv_path, vid_path


def gen_sentence(step_name):
    sentence = f'The action in the video is {step_name}, describe the action.'.format(step_name)
    return sentence                 


def start_end():
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    

    csv_path, vid_path = read_videos_and_step_start_end()
    #print(csv_path, vid_path)
    cos = torch.nn.CosineSimilarity(eps=1e-6)
    
    for i in range(len(vid_path)):
        # read csv for steps
        Path('NIV/start_end_feat/' + vid_path[i].split('/')[-1]).mkdir(parents=True, exist_ok=True)
        steps = genfromtxt(csv_path[i], delimiter=',', dtype=str)[:,0]
        #print(steps)
        previous_tensor = None
        previous_vid_feat = None
        for j in range(len(steps)):
            # read step video
            print(vid_path[i] + '/step' + str(j) + '.mp4')
            
            # start obs
            if os.path.isfile(vid_path[i] + '/step' + str(j) + '_start.mp4'):
                sentence = gen_sentence(steps[j])
                print(sentence)

                video_processor = processor['video']
                conv_mode = "llava_v1"
                
                conv = conv_templates[conv_mode].copy()
                roles = conv.roles
                video_tensor = video_processor(vid_path[i] + '/step' + str(j) + '_start.mp4', return_tensors='pt')['pixel_values']
                print(video_tensor.shape)
                if j > 0:
                    #calc similarity
                    print('cos sim', torch.mean(cos(video_tensor, previous_tensor)))
                previous_tensor = video_tensor
                if type(video_tensor) is list:
                    tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
                else:
                    tensor = video_tensor.to(model.device, dtype=torch.float16)
                key = ['video']
                
                print(f"{roles[1]}: {sentence}")
                inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + sentence
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                
                '''with torch.inference_mode():
                    output_ids = model.generate(
                                 input_ids,
                                 images=[tensor, key],
                                 do_sample=True,
                                 temperature=0.1,
                                 max_new_tokens=1024,
                                 use_cache=True,
                                 stopping_criteria=[stopping_criteria])

                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                print(outputs[:-4], type(outputs))
                with open(vid_path[i]+'/step'+str(j)+'.txt', "w") as text_file:
                    text_file.write(outputs[:-4])'''

            
                with torch.inference_mode():
                    vid_feat = model.encode_videos(tensor)#.cpu().numpy()
                    if j>0:
                        print('cos sim vid feat', torch.mean(cos(vid_feat, previous_vid_feat)))
                previous_vid_feat = vid_feat
                vid_feat = vid_feat.cpu().numpy()
                    #output_words = tokenizer(outputs[:-4], padding=False, truncation=True, return_tensors='pt')
                    #text_feat_last_state = model(**output_words, images=[tensor, key], output_hidden_states=True).hidden_states[0][:,-1,:]#[0][:,-1,:] #.hidden_states()
                #text_feat_last_state = text_feat_last_state.cpu().numpy()
                #np.save(vid_path[i]+'/act_feat_step'+str(j)+'.npy', text_feat_last_state)
                np.save('NIV/start_end_feat/'+vid_path[i].split('/')[-1]+'/vid_feat_step'+str(j)+'_start.npy', vid_feat)
                
            # end obs
            if os.path.isfile(vid_path[i] + '/step' + str(j) + '_end.mp4'):
                sentence = gen_sentence(steps[j])
                print(sentence)
                
                conv = conv_templates[conv_mode].copy()
                roles = conv.roles
                video_tensor = video_processor(vid_path[i] + '/step' + str(j) + '_end.mp4', return_tensors='pt')['pixel_values']
                if type(video_tensor) is list:
                    tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
                else:
                    tensor = video_tensor.to(model.device, dtype=torch.float16)
                key = ['video']
                
                print(f"{roles[1]}: {sentence}")
                inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + sentence
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                
                '''with torch.inference_mode():
                    output_ids = model.generate(
                                 input_ids,
                                 images=[tensor, key],
                                 do_sample=True,
                                 temperature=0.1,
                                 max_new_tokens=1024,
                                 use_cache=True,
                                 stopping_criteria=[stopping_criteria])

                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                print(outputs[:-4], type(outputs))
                with open(vid_path[i]+'/step'+str(j)+'.txt', "w") as text_file:
                    text_file.write(outputs[:-4])'''

            
                with torch.inference_mode():
                    vid_feat = model.encode_videos(tensor).cpu().numpy()                    
                    #output_words = tokenizer(outputs[:-4], padding=False, truncation=True, return_tensors='pt')
                    #text_feat_last_state = model(**output_words, images=[tensor, key], output_hidden_states=True).hidden_states[0][:,-1,:]#[0][:,-1,:] #.hidden_states()
                #text_feat_last_state = text_feat_last_state.cpu().numpy()
                #np.save(vid_path[i]+'/act_feat_step'+str(j)+'.npy', text_feat_last_state)
                np.save('NIV/start_end_feat/'+vid_path[i].split('/')[-1]+'/vid_feat_step'+str(j)+'_end.npy', vid_feat)

if __name__ == '__main__':
    start_end()

