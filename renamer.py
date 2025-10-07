import os

for i in range(1,22):
    os.rename(src=f"rename_me/{i}_camera-1_0_synced.csv",dst=f"rename_me_output/{i}_Empty_Cage_Left_Sync.avi")
