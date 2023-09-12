# dift-video (WIP)

Simple wrapper to apply SD Feature Extractors to find correspondences in video.

Can 
1. load some frames
2. Extract their SD features
3. Find correspondences using cos-similarity
4. (soon) Save correspondences as json files for each frame (for each frame we'll store an array of keypoints like: `{kpid: string, x: number, y: number, frame: number}[]`)
5. (soon) visualize correspondences drawing they keypoints on each frame and optionally add trails


