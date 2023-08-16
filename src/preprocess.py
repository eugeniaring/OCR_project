import numpy as np
import pandas as pd
import cv2
import tqdm
from glob import glob

image_path = 'data/images/'
gt_path = 'data/labels/'

df = pd.read_parquet('data/labels/annot.parquet')
df['text_len'] = df.utf8_string.str.len()
print(df.head())



# long_image_ids = list(df[df.text_len>=5].image_id.unique())
# long_image_ids = long_image_ids[:1000]
# df = df[df.image_id.isin(long_image_ids[:1000])]

train_image_paths = glob(image_path)
# train_ids = [str(new_file.split('/')[-1].split('.')[0]) for new_file in train_image_paths]
# long_image_ids = list(set(long_image_ids).intersection(set(train_ids)))

# print(long_image_ids[:5])
# train_image_paths = ['{}{}.jpg'.format(image_path,str(el)) for el in long_image_ids]
# print(train_image_paths[:5])

X_final = []
Y_final = []
grid_h = 16
grid_w = 16
img_w = 512
img_h = 512

train_image_paths = train_image_paths[:1500]


for idx in tqdm.tqdm(range(len(train_image_paths))):
    new_file = train_image_paths[idx]
    x = cv2.imread(new_file)
    print(new_file)

    img_id = str(new_file.split('/')[-1].split('.')[0])
    sub_df = df[df.image_id==img_id][['bbox','utf8_string']]
    print('image {} has {} objects'.format(img_id,len(sub_df)))

    if len(sub_df)!=0:
        x_sl = 512/x.shape[1]
        y_sl = 512/x.shape[0]
        img = cv2.resize(x,(512,512))
        Y = np.zeros((grid_h,grid_w,1,5))
        X_final.append(img)
        for index, r in sub_df.iterrows():
            bb,strr = r['bbox'],r['utf8_string']
            xmin = int(bb[0])*x_sl
            xmax = int(bb[2])*x_sl
            ymin = int(bb[1])*y_sl
            ymax = int(bb[3])*y_sl

            w = (xmax - xmin)/img_w
            h = (ymax - ymin)/img_h
            
            x = ((xmax + xmin)/2)/img_w
            y = ((ymax + ymin)/2)/img_h
            x = x * grid_w
            y = y * grid_h
            
            Y[int(y),int(x),0,0] = 1
            Y[int(y),int(x),0,1] = x - int(x)
            Y[int(y),int(x),0,2] = y - int(y)
            Y[int(y),int(x),0,3] = w
            Y[int(y),int(x),0,4] = h

        Y_final.append(Y)

X = np.array(X_final)
X_final = []
Y = np.array(Y_final)
Y_final = []

X = (X - 127.5)/127.5

np.save('data/X.npy',X)
np.save('data/Y.npy',Y)