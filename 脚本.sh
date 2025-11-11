python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from utils import general_utils; print('Utils imported successfully')"
------
gaussian_splatting
PyTorch: 2.2.0+cu121, CUDA: True
Utils imported successfully
-------
cd ~/Desktop
python -m zipfile -e SegAnyGAussians.zip .
cd SegAnyGAussians-2
mkdir -p data/train_dataset/book_scene/colmap/images
ffmpeg -i ../book.mp4 -r 3 -q:v 2 data/train_dataset/book_scene/colmap/images/frame_%04d.jpg
--------------
(gaussian_splatting) bygpu@G39126:~/Desktop/SegAnyGAussians-2$ ffmpeg -i ../book.mp4 -r 3 -q:v 2 data/train_dataset/book_scene/colmap/images/frame_%04d.jpg
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '../book.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 1
    compatible_brands: isommp41mp42
    creation_time   : 2025-11-11T09:44:37.000000Z
    copyright       : 
    copyright-eng   : 
  Duration: 00:01:15.53, start: 0.000000, bitrate: 1349 kb/s
  Stream #0:0(und): Video: hevc (Main) (hvc1 / 0x31637668), yuv420p(tv, bt709), 720x1280, 1295 kb/s, 30 fps, 30 tbr, 600 tbn, 600 tbc (default)
    Metadata:
      creation_time   : 2025-11-11T09:44:37.000000Z
      handler_name    : Core Media Video
      vendor_id       : [0][0][0][0]
  Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 48 kb/s (default)
    Metadata:
      creation_time   : 2025-11-11T09:44:37.000000Z
      handler_name    : Core Media Audio
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (hevc (native) -> mjpeg (native))
Press [q] to stop, [?] for help
[swscaler @ 0x5b306117d1c0] deprecated pixel format used, make sure you did set range correctly
Output #0, image2, to 'data/train_dataset/book_scene/colmap/images/frame_%04d.jpg':
  Metadata:
    major_brand     : mp42
    minor_version   : 1
    compatible_brands: isommp41mp42
    copyright-eng   : 
    copyright       : 
    encoder         : Lavf58.76.100
  Stream #0:0(und): Video: mjpeg, yuvj420p(pc, bt709, progressive), 720x1280, q=2-31, 200 kb/s, 3 fps, 3 tbn (default)
    Metadata:
      creation_time   : 2025-11-11T09:44:37.000000Z
      handler_name    : Core Media Video
      vendor_id       : [0][0][0][0]
      encoder         : Lavc58.134.100 mjpeg
    Side data:
      cpb: bitrate max/min/avg: 0/0/200000 buffer size: 0 vbv_delay: N/A
frame=  228 fps= 36 q=2.0 Lsize=N/A time=00:01:16.00 bitrate=N/A dup=0 drop=2038 speed=11.9x    
video:34380kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
-----------------------
# 用2fps重新提取（约150张）
ffmpeg -i ../book.mp4 -r 2 -q:v 2 data/train_dataset/book_scene/colmap/images/frame_%04d.jpg
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '../book.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 1
    compatible_brands: isommp41mp42
    creation_time   : 2025-11-11T09:44:37.000000Z
    copyright       : 
    copyright-eng   : 
  Duration: 00:01:15.53, start: 0.000000, bitrate: 1349 kb/s
  Stream #0:0(und): Video: hevc (Main) (hvc1 / 0x31637668), yuv420p(tv, bt709), 720x1280, 1295 kb/s, 30 fps, 30 tbr, 600 tbn, 600 tbc (default)
    Metadata:
      creation_time   : 2025-11-11T09:44:37.000000Z
      handler_name    : Core Media Video
      vendor_id       : [0][0][0][0]
  Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 48 kb/s (default)
    Metadata:
      creation_time   : 2025-11-11T09:44:37.000000Z
      handler_name    : Core Media Audio
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (hevc (native) -> mjpeg (native))
Press [q] to stop, [?] for help
[swscaler @ 0x5dacef03b1c0] deprecated pixel format used, make sure you did set range correctly
Output #0, image2, to 'data/train_dataset/book_scene/colmap/images/frame_%04d.jpg':
  Metadata:
    major_brand     : mp42
    minor_version   : 1
    compatible_brands: isommp41mp42
    copyright-eng   : 
    copyright       : 
    encoder         : Lavf58.76.100
  Stream #0:0(und): Video: mjpeg, yuvj420p(pc, bt709, progressive), 720x1280, q=2-31, 200 kb/s, 2 fps, 2 tbn (default)
    Metadata:
      creation_time   : 2025-11-11T09:44:37.000000Z
      handler_name    : Core Media Video
      vendor_id       : [0][0][0][0]
      encoder         : Lavc58.134.100 mjpeg
    Side data:
      cpb: bitrate max/min/avg: 0/0/200000 buffer size: 0 vbv_delay: N/A
frame=  153 fps= 25 q=2.0 Lsize=N/A time=00:01:16.50 bitrate=N/A dup=0 drop=2113 speed=12.4x    
video:22664kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
-----------------------------------

# 进入数据目录
cd data/train_dataset/book_scene/colmap

# 1. 特征提取（最关键的步骤）
colmap feature_extractor --database_path database.db --image_path images --ImageReader.single_camera 1

# 2. 特征匹配
colmap exhaustive_matcher --database_path database.db

# 3. 映射重建
colmap mapper --database_path database.db --image_path images --output_path sparse

cd ../../../..

# 开始训练高斯泼溅模型
python train_scene.py -s data/train_dataset/book_scene/colmap --iterations 10000
--------------------------
Optimizing 
Output folder: ./output/b8b138cc-5 [11/11 22:49:36]
Allow Camera Principle Point Shift: False [11/11 22:49:36]
Reading camera 153/153 [11/11 22:49:37]
Converting point3d.bin to .ply, will happen only the first time you open the scene. [11/11 22:49:37]
Loading Training Cameras [11/11 22:49:38]
Loading Test Cameras [11/11 22:49:40]
Number of points at initialisation :  84080 [11/11 22:49:40]
Training progress:  70%|████████████████████████████████████████████████████████████████████████████████████████▉                                      | 7000/10000 [04:46<02:42, 18.42it/s, Loss=0.0368998]
[ITER 7000] Saving Gaussians [11/11 22:54:27]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [07:53<00:00, 21.10it/s, Loss=0.0370931]

[ITER 10000] Saving Gaussians [11/11 22:57:35]

Training complete. [11/11 22:57:49]
--------------------------

# 训练对比特征
python train_contrastive_feature.py -m output/b8b138cc-5 --iterations 3000

# 训练完成后会自动生成 contrastive_feature_point_cloud.ply