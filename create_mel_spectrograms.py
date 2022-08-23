import argparse
#import sys
#sys.path.append('tacotron2/')
#from tacotron2.hparams import create_hparams
from hparams import create_hparams
#from tacotron2.layers import TacotronSTFT
from layers import TacotronSTFT
import json
import pandas as pd
import os
from io import BytesIO
import torchaudio
import torch
import soundfile as sf
import numpy as np
import io
import tqdm
import matplotlib.pyplot as plt

def save_spec(spec, output_filepath):
   fig, axs = plt.subplots(1, 1)
   axs.set_title("Spectrogram (db)")
   axs.set_ylabel('Freq')
   axs.set_xlabel("frame")
   im = axs.imshow(spec.squeeze(), origin="lower")
   fig.colorbar(im, ax=axs)
   plt.savefig(output_filepath)


def read_json(json_path):
    '''
    Функция отвечает за считывание файлов manifest.json
    '''
    dataset_type = json_path.split('_')[-1].replace('.json', '')
    with open(json_path, encoding='utf-8') as f:
        cond = "[" + f.read().replace("}\n{", "},\n{") + "]"
        json_data = json.loads(cond)
        for item in json_data:
            item['dataset_type'] = dataset_type
    return json_data

def read_waveform(audio_filepath, target_sr):
    # Считываем аудио-данные и частоту дискретизации файла (.flac, 44100Hz, pcm-f)
    #flac_data, sample_rate = librosa.load(load_flac_path)
    waveform, sr = torchaudio.load(audio_filepath)

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        sr = target_sr
    return waveform


def convert_waveform_to_mel(hp, waveform):
    # Формируем мел-спектрограмму
    # melspec_1 = librosa.feature.melspectrogram(y=flac_data,sr=sample_rate)
    '''
    melspec_1 = torchaudio.transforms.MelSpectrogram(n_fft=hp['filter_length'], sample_rate=hp['sampling_rate'], n_mels=hp['n_mel_channels'])(waveform)

    # Отсекаем слишком большие спектрограммы
    # для nvidia tesla t4 16gb с размером спектрограммы <1000 получалось установить размер батча=64
    # иначе может вылетать ошибка pytorch о переполнении памяти gpu, и придется уменьшать размер батча
    # зависит от gpu на которой будут происходить вычисления
    if melspec_1.shape[1] >= 1000:
        return False

    # Формируем новое аудио для записи в память
    #audio = librosa.feature.inverse.mel_to_audio(melspec_1, sr=sample_rate)
    spec = torchaudio.transforms.InverseMelScale(sample_rate=hp['sampling_rate'], n_stft=int(hp['filter_length'])//2+1, n_mels=hp['n_mel_channels'], f_max=hp['mel_fmax'], f_min=hp['mel_fmin'])(melspec_1)
    inv_waveform = torchaudio.transforms.GriffinLim(n_fft=hp['filter_length'])(spec)

    # Буфер памяти (что-бы не сохранять локально)
    buffer_ = BytesIO()

    # Запись файла с другими параметрами, нежели были изначально
    # Необходимо, т.к. используется вокодер обученный на аудио с частотой дискретизации = 22050Hz
    # Так-же метод write модуля scipy считывает только wav формат
    # (считанные данные из flac файлов библиотеками librosa и soundfile, почему-то некорректно преобразовывались в
    # mel-спектрограммы модулем stft)

    #write(buffer_, sr, audio)
    torchaudio.save(buffer_, inv_waveform, hp['sampling_rate'], format="wav")
    buffered_waveform = buffer_.getvalue()
    buffer_.close()

    # Считываем аудио-данные и частоту дискретизации файла (.wav, 22050Hz, pcm-s)
    buf_data, sr = sf.read(file=io.BytesIO(buffered_waveform), dtype='float32')
    # Преобразовываем в тензор
    floated_data = torch.FloatTensor(buf_data)

    floated_data = torch.FloatTensor(buf_data)
    '''
    floated_waveform = torch.FloatTensor(waveform)

    # Формирование мел-спектрограммы
    norm_waveform = floated_waveform / hp['max_wav_value']
    norm_waveform = norm_waveform.unsqueeze(0)
    norm_waveform = torch.autograd.Variable(norm_waveform, requires_grad=False)
    spec = torchaudio.transforms.Spectrogram(n_fft=hp['filter_length'], power=2.0, normalized=True)(norm_waveform)
    #melspec_2 = torchaudio.transforms.MelSpectrogram(n_fft=hp['filter_length'], sample_rate=hp['sampling_rate'], n_mels=hp['n_mel_channels'])(norm_waveform)
    n_fft = int(hp['filter_length'])
    melspec = torchaudio.transforms.MelScale(n_mels=hp['n_mel_channels'], sample_rate=hp['sampling_rate'], f_min=hp['mel_fmin'], f_max=hp['mel_fmax'], n_stft=n_fft//2+1)(spec)

    #spec = torchaudio.transforms.InverseMelScale(sample_rate=hp['sampling_rate'], n_stft=n_fft//2+1, n_mels=hp['n_mel_channels'], f_max=hp['mel_fmax'], f_min=hp['mel_fmin'])(melspec)
    #melspec_2 = taco_stft.mel_spectrogram(norm_waveform)
    melspec = torch.squeeze(melspec, 0)

    return spec, melspec


def flac_to_mel(taco_stft_fn, hp, input_filepath, output_filepath, dataset_type):
    '''
    Функция формирует мел-спектрограмму из аудио-файла и сохраняет её
    '''
    waveform = read_waveform(input_filepath, target_sr=hp['sampling_rate'])
    #spec, melspec = convert_waveform_to_mel(hp, waveform)
    waveform_norm = waveform / hp['max_wav_value']
    waveform_norm = waveform_norm.squeeze()
    waveform_norm = torch.autograd.Variable(waveform_norm, requires_grad=False)
    waveform_norm = waveform_norm.unsqueeze(0)
    melspec = taco_stft_fn.mel_spectrogram(waveform_norm)
    melspec = torch.squeeze(melspec, 0)
    # Сохранение файла
    #save_spec(melspec, output_filepath + 'png')
    torch.save(melspec, output_filepath)


def execute_process(hp, hifitts_path, output_path):

    # Формирование единого датафрейма по всем manifest-файлам .json
    manifests = [manifest for manifest in os.listdir(hifitts_path) if 'manifest' in manifest]
    manifest_paths = [f'{hifitts_path}/{manifest}' for manifest in manifests]
    manifest_jsons = [read_json(manifest_path) for manifest_path in manifest_paths]
    manifest_dfs = [pd.DataFrame(manifest_json) for manifest_json in manifest_jsons]
    manifests_df = pd.concat(manifest_dfs, axis=0)

    df = manifests_df.reset_index(drop=True).copy()

    # Формирование колонки с нормализованным id диктора (от 0 до 9)
    df['reader_id'] = df['audio_filepath'].apply(lambda x: x.split('/')[1].split('_')[0])
    readers_list = [reader_id for reader_id in df.reader_id.unique()]
    readers_dict = {reader_id: str(readers_list.index(reader_id)) for reader_id in readers_list}
    df['reader_id_norm'] = df['reader_id'].apply(lambda x: readers_dict[x])

    # Формирование строки текстового файла по которому модель будет обучаться/валидироваться
    df['mel_path'] = 'mels/' + df.index.astype('string') + '_' + df['dataset_type'] + '_' + df['reader_id']
    df['txt_line'] = df['mel_path'] + '|' + df['text_normalized'] + '|' + df['reader_id_norm'] + '\n'

    # Оставляем только необходимые колонки
    df = df[['dataset_type', 'reader_id', 'reader_id_norm', 'text', 'audio_filepath', 'mel_path', 'txt_line']]

    # Оставляем только тестовую и тренеровочную выборки
    df = df[df['dataset_type'] != 'dev']

    # Создание директории для записи файлов
    os.makedirs(output_path, exist_ok = True)
    os.makedirs(os.path.join(output_path, 'mels'), exist_ok = True)

    tmp_df = df.copy()
    tmp_df = df.sample(n=1000)
    tmp_df['audio_filepath'] = hifitts_path + '/' + tmp_df['audio_filepath']
    tmp_df['mel_path'] = output_path + '/' + tmp_df['mel_path']

    #with open('./hifitts/' + dataset_type + '.txt', 'w') as f:
    #    f.write('')

    taco_stft = TacotronSTFT(
        hp['filter_length'],
        hp['hop_length'],
        hp['win_length'],
        hp['n_mel_channels'],
        hp['sampling_rate'],
        hp['mel_fmin'],
        hp['mel_fmax']
    )

    for index, row in tqdm.tqdm(tmp_df.iterrows(), total=tmp_df.shape[0]):

        filepath, text, speaker_id = row['txt_line'].split('|')
        #full_filepath = row['audio_filepath'].replace('.flac', '.wav')
        full_filepath = row['audio_filepath']
        if not (os.path.exists(full_filepath)):
           print(full_filepath)
           continue

        flac_to_mel(
            taco_stft,
            hp,
            full_filepath,
            row['mel_path'],
            row['dataset_type']
        )
        # Записываем информационную строку о текущем элементе в тексовый файл для обучения/валидации модели
        with open('./hifitts/' + row['dataset_type'] + '.txt', 'a') as f:
            f.write(row['txt_line'])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default='./')
  parser.add_argument('--input_dir', default='./hi_fi_tts_v0', help='HiFi Dataset path')
  parser.add_argument('--output_dir', default='./hifitts', help='Name of the output directory of mel files')
  args = parser.parse_args()

  dataset_path = os.path.join(args.base_dir, args.input_dir)
  output_path = os.path.join(args.base_dir, args.output_dir)

  hparams = create_hparams()

  execute_process(hparams, dataset_path, output_path)

if __name__ == "__main__":
  main()
