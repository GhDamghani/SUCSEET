from transformers import AutoProcessor, HubertModel
from os.path import join, exists
from os import mkdir
import numpy as np
from scipy.io.wavfile import read
from tqdm import tqdm

processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

result_dir = 'results'
features_dir = join('..', 'SingleWordProductionDutch-main', 'features')
if not exists(result_dir):
    mkdir(result_dir)

eeg_sr = 1024

for p_ind in tqdm(range(1, 11)):
    audio_sr, sound = read(join(features_dir, f'sub-{p_ind:02}_orig_audio.wav'))
    sound = sound/32767
    words_arr = np.load(join(features_dir, f'sub-{p_ind:02}_procWords.npy'))
    words = np.unique(words_arr)
    words = words[np.where(words != b'')]
    sub_dir = f'sub-{p_ind:02}'
    if not exists(join(result_dir, sub_dir)):
        mkdir(join(result_dir, sub_dir))
    for word_i, word in tqdm(enumerate(words)):
        word_ind = np.where(words_arr == word)[0][[0, -1]]
        word_ind_start = np.int32(np.round_(word_ind[0]/eeg_sr * audio_sr))
        word_ind_stop = np.int32(np.round_((word_ind[1]+1)/eeg_sr * audio_sr))
        word_audio = sound[word_ind_start:word_ind_stop]
        input_values = processor(word_audio, return_tensors="pt", sampling_rate=audio_sr).input_values  # Batch size 1
        hidden_states = model(input_values).last_hidden_state.detach().numpy()
        np.save(join(result_dir, sub_dir, f'{word_i:03}_{word.decode("ascii")}'), hidden_states)



