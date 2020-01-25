import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
import copy


# ## Class for generating features
# In the hidden cell below you can find a class for feature generation.

class FeatureGenerator(object):
    def __init__(self, n_jobs=1, df=None, dataset: str = 'train'):
        self.n_jobs = n_jobs
        self.df = df
        self.dataset = dataset

    def read_chunks(self):
        for id, user_sample in self.df.groupby('installation_id', sort=False):
            yield id, user_sample

    def get_features(self, row):
        return self.features(row)

    def features(self, id, user_sample):
        user_data = []

        accuracy_mapping = {0: 0, 1: 3, 0.5: 2}

        user_stats = defaultdict(int)
        user_stats['installation_id'] = user_sample['installation_id'].unique()[0]
        user_stats['world'] = user_sample['world'].unique()[0]
        user_stats['timestamp'] = user_sample['timestamp'].unique()[0]

        temp_dict = defaultdict(int)
        another_temp_dict = {}
        another_temp_dict['durations'] = []
        another_temp_dict['all_durations'] = []
        another_temp_dict['durations_with_attempts'] = []
        another_temp_dict['mean_action_time'] = []
        title_data = defaultdict(dict)

        for i, session in user_sample.groupby('game_session', sort=False):
            user_stats['last_ass_session_game_time'] = another_temp_dict['durations'][-1] if len(another_temp_dict['durations']) > 0 else 0
            user_stats['last_session_game_time'] = another_temp_dict['all_durations'][-1] if len(another_temp_dict['all_durations']) > 0 else 0

            # calculate some user_stats and append data
            if session['attempt'].sum() > 0 or self.dataset == 'test':
                user_stats['session_title'] = session['title'].values[0]
                accuracy = np.nan_to_num(session['correct'].sum() / session['attempt'].sum())
                if accuracy in accuracy_mapping.keys():
                    user_stats['accuracy_group'] = accuracy_mapping[accuracy]
                else:
                    user_stats['accuracy_group'] = 1
                user_stats['accumulated_accuracy_group'] = temp_dict['accumulated_accuracy_group'] / user_stats['counter'] if user_stats['counter'] > 0 else 0
                temp_dict['accumulated_accuracy_group'] += user_stats['accuracy_group']
                user_data.append(copy.copy(user_stats))

            user_stats[session['type'].values[-1]] += 1
            user_stats['accumulated_correct_attempts'] += session['correct'].sum()
            user_stats['accumulated_uncorrect_attempts'] += session['attempt'].sum() - session['correct'].sum()
            event_code_counts = session['event_code'].value_counts()
            for i, j in zip(event_code_counts.index, event_code_counts.values):
                user_stats[i] += j

            temp_dict['assessment_counter'] += 1
            if session['title'].values[-1] in title_data.keys():
                pass
            else:
                title_data[session['title'].values[-1]] = defaultdict(int)

            title_data[session['title'].values[-1]]['duration_all'] += session['game_time'].values[-1]
            title_data[session['title'].values[-1]]['counter_all'] += 1

            user_stats['duration'] = (session.iloc[-1,2] - session.iloc[0,2]).seconds
            if session['type'].values[0] == 'Assessment' and (len(session) > 1 or self.dataset == 'test'):
                another_temp_dict['durations'].append(user_stats['duration'])
                accuracy = np.nan_to_num(session['correct'].sum() / session['attempt'].sum())
                user_stats['accumulated_accuracy_'] += accuracy
                user_stats['counter'] += 1
                if user_stats['counter'] == 0:
                    user_stats['accumulated_accuracy'] = 0
                else:
                    user_stats['accumulated_accuracy'] = user_stats['accumulated_accuracy_'] / user_stats['counter']

                accuracy = np.nan_to_num(session['correct'].sum() / session['attempt'].sum())

                if accuracy in accuracy_mapping.keys():
                    user_stats[accuracy_mapping[accuracy]] += 1
                else:
                    user_stats[1] += 1

                user_stats['accumulated_actions'] += len(session)

                if session['attempt'].sum() > 0:
                    user_stats['sessions_with_attempts'] += 1
                    another_temp_dict['durations_with_attempts'].append(user_stats['duration'])

                if session['correct'].sum() > 0:
                    user_stats['sessions_with_correct_attempts'] += 1
                    
                user_stats['title_duration'] = title_data[session['title'].values[-1]]['duration']
                user_stats['title_counter'] = title_data[session['title'].values[-1]]['counter']
                user_stats['title_mean_duration'] = user_stats['title_duration'] / user_stats['title_mean_duration']  if user_stats['title_mean_duration'] > 0 else 0

                user_stats['title_duration_all'] = title_data[session['title'].values[-1]]['duration_all']
                user_stats['title_counter_all'] = title_data[session['title'].values[-1]]['counter_all']
                user_stats['title_mean_duration_all'] = user_stats['title_duration_all'] / user_stats['title_mean_duration_all']  if user_stats['title_mean_duration_all'] > 0 else 0
                
                title_data[session['title'].values[-1]]['duration'] += session['game_time'].values[-1]
                title_data[session['title'].values[-1]]['counter'] += 1

            elif (len(session) > 1 or self.dataset == 'test'):
                another_temp_dict['all_durations'].append(user_stats['duration'])


            if user_stats['duration'] != 0:
                temp_dict['nonzero_duration_assessment_counter'] += 1
            user_stats['duration_mean'] = np.mean(another_temp_dict['durations'])
            user_stats['duration_attempts'] = np.mean(another_temp_dict['durations_with_attempts'])

            # stats from all sessions
            user_stats['all_duration_mean'] = np.mean(another_temp_dict['all_durations'])
            user_stats['all_accumulated_actions'] += len(session)
            user_stats['mean_action_time'] = np.mean(another_temp_dict['mean_action_time'])
            another_temp_dict['mean_action_time'].append(session['game_time'].values[-1] / len(session))


        if self.dataset == 'test':
            user_data = [user_data[-1]]

        return user_data

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.features)(id, user_sample)
                                                                for id, user_sample in self.read_chunks())
        for r in res:
            for r1 in r:
                feature_list.append(r1)
        return pd.DataFrame(feature_list)

# ### Preparing the data
def preprocess(train, train_labels):
    # rows with attempts
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    train['attempt'] = 0
    train.loc[(train['title'] == 'Bird Measurer (Assessment)') & (train['event_code'] == 4110),
        'attempt'] = 1
    train.loc[(train['type'] == 'Assessment') &
        (train['title'] != 'Bird Measurer (Assessment)')
        & (train['event_code'] == 4100), 'attempt'] = 1

    train['correct'] = None
    train.loc[(train['attempt'] == 1) & (train['event_data'].str.contains('"correct":true')), 'correct'] = True
    train.loc[(train['attempt'] == 1) & (train['event_data'].str.contains('"correct":false')), 'correct'] = False

    train = train.loc[train['installation_id'].isin(train_labels['installation_id'].unique())]

    fg = FeatureGenerator(n_jobs=2, df=train)
    data = fg.generate()
    data = data.fillna(0)

    return data

if __name__ == "__main__":

    path = 'C:/Work/gitsrc/Kaggle/data-science-bowl-2019'
    specs = pd.read_csv(f'{path}/specs.csv')
    sample_submission = pd.read_csv(f'{path}/sample_submission.csv')
    train_labels = pd.read_csv(f'{path}/train_labels.csv')
    test = pd.read_csv(f'{path}/test.csv')
    train = pd.read_csv(f'{path}/train.csv')

    train.head()

    train_labels.head()

    specs.head()

    print(f'Rows in train data: {train.shape[0]}')
    print(f'Rows in train labels: {train_labels.shape[0]}')
    print(f'Rows in specs data: {specs.shape[0]}')

    data = preprocess(train, train_labels)
    data.head()
    data.to_csv('intermediate.csv', index=False, header = True)
