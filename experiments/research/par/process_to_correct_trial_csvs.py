df = pd.read_csv('var_sample_trials_m24/OND/activity_recognition/OND.0.10001.6438158_single_df.csv')

# Check columns and the annotation name's unique values. Found for 1st, only
# perspective and location were incorrect.
df.columns
df['annotation_value_1'].unique()
df['annotation_name_1'].unique()
df['annotation_name_2'].unique()
df['annotation_name_3'].unique()
df['annotation_name_4'].unique()
df['annotation_name_1'].unique()
df['annotation_name_1'].unique()

# Rename perspective and location columns
df.rename(columns={'annotation_value_1':df['annotation_name_1'].unique()[0].lower()}, inplace=True)

df.rename(columns={'annotation_value_2':df['annotation_name_2'].unique()[0].lower()}, inplace=True)

# rm unnecessary columns
df.drop(columns=['annotation_name_1', 'annotation_name_2'], inplace=True)

# Rename the rest to what they should be
df.rename(columns={'annotation_name_3': 'relation_type1'}, inplace=True)
df.rename(columns={'annotation_value_3': 'relation1'}, inplace=True)
df.rename(columns={'annotation_name_4': 'relation_type2'}, inplace=True)
df.rename(columns={'annotation_value_4': 'relation2'}, inplace=True)

df.to_csv('var_sample_trials_m24/OND/activity_recognition/corrections/OND.0.10001.6438158_single_df.csv', index=False)
