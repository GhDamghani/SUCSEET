from transformers import AutoProcessor, HubertModel
from os.path import join, exists
from os import mkdir
import numpy as np
from scipy.io.wavfile import read

processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

result_dir = 'results'
if not exists(result_dir):
    mkdir(result_dir)

for p_ind in range(1, 11):
    sr, sound = read(join('..', 'SingleWordProductionDutch-main', 'features', f'sub-{p_ind:02}_orig_audio.wav'))
    sound = sound/32767
    input_values = processor(sound, return_tensors="pt", sampling_rate=sr).input_values  # Batch size 1
    hidden_states = model(input_values[:, :input_values.shape[1]//1000]).last_hidden_state.detach().numpy()
    np.save(join(result_dir, f'sub-{p_ind:02}_hidden_units'), hidden_states)
