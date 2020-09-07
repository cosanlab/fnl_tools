import numpy as np
import pandas as pd
from nltools.data import Adjacency

def create_long_character_annotation(annotations, movie_len=45*60+17, character_list=None):
    if character_list is not None:
        dm = pd.DataFrame(data=np.zeros((movie_len,len(character_list))),columns=character_list)
        for c in character_list:
            for t in np.where(annotations[c]==True)[0]:
                start = np.floor(annotations.loc[t,'Begin Time - ss.msec'])
                stop = np.ceil(annotations.loc[t,'End Time - ss.msec'])
                dm.loc[start:stop,c] = 1
        return dm
    else:
        raise ValueError('Must specify which characters to return.')

def create_long_annotation(annotations, movie_len=45*60+17, annotation_column=None):
    if annotation_column is not None:
        annotation_items = annotations[annotation_column].unique()
        dm = pd.DataFrame(data=np.zeros((movie_len,len(annotation_items))),columns=annotation_items)
        for i in annotation_items:
            for t in np.where(annotations[annotation_column]==i)[0]:
                start = np.floor(annotations.loc[t,'Begin Time - ss.msec'])
                stop = np.ceil(annotations.loc[t,'End Time - ss.msec'])
                dm.loc[start:stop,i] = 1
        return dm
    else:
        raise ValueError('Must specify which annotation column to return.')

def convert_episode_to_adjacency(data, episode, dimension='likePair', char_list=None):
    if char_list is None:
        char_list = ['eric', 'tami', 'brian', u'matt','lyla', 'tim', 'tyra',
                     'landry', 'jason', 'julie', 'lorraine','buddy', 'billy',
                     'extra']
    sub_list = data['subject_id'].unique()
    all_dat = []
    for s in sub_list:
        s_dat = data.loc[(data['dimension']==dimension) & (data['episode']==episode) & (data['subject_id']==s)]
        s_square_dat = pd.DataFrame(np.ones((len(char_list),len(char_list)))*100,index=char_list,columns=char_list)
        for i in char_list:
            for j in char_list:
                if i is not j:
                    s_square_dat.loc[i,j] = s_dat.loc[(s_dat['char1']==i) & (s_dat['char2']==j),'rating'].values
        all_dat.append(s_square_dat)
    return Adjacency(data=all_dat,matrix_type='directed')

def create_scene_annotation(annotations, n_tr = 1364, tr=2.0):
    dm = pd.DataFrame({'Scene':np.zeros((n_tr))})
    for i,s in annotations.iterrows():
        if i < annotations.shape[0]-1:
            start = np.floor(s['Onset']/tr)
            stop = np.floor(annotations.loc[i+1,'Onset']/tr)
        elif i == annotations.shape[0]-1:
            stop = n_tr
        dm.loc[start:stop,'Scene'] = s['Scene']
    return dm

def create_emotion_df(dat, name='Joy'):
    '''Create Dataframe from Emotion Rating Dataset'''
    dat = pd.DataFrame(columns=range(1364*2))
    for i in ratings[name]:
        for j in ratings[name][i]:
            dat.loc[int(float(i)),int(float(j))] = ratings[name][i][j]
    dat.replace(-1, 0, inplace=True)
    return dat


def convert_data_from_database(file_name, output_file_name):
    '''Simple function to convert data into json object.
    Written by Nathan Greenstein 
    '''
    data = {}

    with open(file_name) as csvFile:
        reader = csv.DictReader(csvFile)
        for row in reader:
            second = round(float(row["time"]))
            ratings = json.loads(row["emotion"])
            for emotion, rating in ratings.items():
                thisEmotion = data.get(emotion, {})
                thisParticipant = thisEmotion.get(int(row["participant"]), {})
                thisParticipant[second] = int(rating)
                thisEmotion[int(row["participant"])] = thisParticipant
                data[emotion] = thisEmotion

    dump = json.dumps(data)
    fh = open(output_file_name,"w")
    fh.write(dump)
    fh.close()
    return json.dumps(data)
