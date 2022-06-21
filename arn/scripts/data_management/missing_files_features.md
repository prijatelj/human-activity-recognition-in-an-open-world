Update on expected versus missing files in each dataset's split for both X3D and TimeSformer

Kitware's features:
    First slack message on missing features:
        https://cvrl.slack.com/archives/C0158MV58QH/p1652070508745489
    First logs of missing files in gdrive:
        https://drive.google.com/drive/folders/1IrmceQQhuokfyE-g-bWw3Vdf7dLD8Sc3?usp=sharing

Date: 2022-06-15
missing logs w/ file paths saved to gdrive:
    x3d:
        https://drive.google.com/drive/folders/1ZeiVExc5YvvLCT9stg-PrvhAl10-Crab?usp=sharing
    timesformer:
        https://drive.google.com/drive/folders/1FRE6l6K-izkNzx750i1dbrEEymr5JjSv?usp=sharing
Note there is overlapping files across dsets.

- X3D: x3d2.zip from June 14
    - K unified: 840,213
        - missing 60173
    - K4
        - train: 219,782
            - missing 1411
        - val: 18,035
            - missing 19
        - test: 35,357
            - missing 100
    - K6
        - train: 371354
            - missing 13025
        - val: 28318
            - missing 13626
        - test: 56618
            - missing 25521
    - K7: Missing more than the first time!
        - train: 544823
            - missing 27533
        - val: 34178
            - missing 1064
- TimeSformer: timesformer.zip from Jun 15
    - K unified: 840,213
        - missing 60203
        - 3 EOFErrors:
            - 37/Vcc_Iyw4iAw_000043_000053_feat.pt
            - 54/JAiSbXzEtYo_000047_000057_feat.pt
            - 55/miMQsHXMCjY_000066_000076_feat.pt
    - K4
        - train: 219,782
            - missing 1411
        - val: 18,035
            - missing 19
        - test: 35,357
            - missing 100
    - K6
        - train: 371354
            - missing 13031
        - val: 28318
            - missing 13626
        - test: 56618
            - missing 25523
    - K7
        - train: 544823
            - missing 27548
        - val: 34178
            - missing 1064
