# Dense Visible Watermark Removal with Progressive Feature Propagation
This paper has been accepted by ICIG 2023.
## Dataset
- Due to copyright infringement concerns, we regretfully cannot provide the relevant dataset. However, we can provide a general guideline on the process of creating such datasets, as follows:
  1. Randomly sample images from each of the four major digital stock photo platforms as evaluation samples. The sampled images from each platform should include various styles, such as photography and digital illustrations, to ensure the integrity of multiple features, including texture and color characteristics, in densely visible watermark embedding areas.
  2. Search for and download the official logo images used by the respective digital stock platforms from their official websites. Based on the specific characteristics of each platform's densely visible watermark format, replicate watermark samples estimated at a size of 512×512 pixels. Aim to maintain consistency in terms of the style, tilt, and color of the embedded watermark as closely as possible with the officially embedded densely visible watermark.
  3. Embed the corresponding estimated watermark samples into regions of authentic samples without watermarks multiple times, using different levels of transparency. The initial embedding intensity starts at 0.5 (using a quantization scale of 0 to 1 for embedding intensity), and the step size for increasing or decreasing the embedding intensity is set to 0.02. Through iterative comparisons between the embedded regions of the estimated watermark and the officially embedded watermark, determine the maximum and minimum estimated values for embedding intensity. Create the relevant dataset based on this range.
## Environment
- git clone https://github.com/HyuuHyuuNeko/PVWRNet.git
- Install dependencies: pip install -r requirements.txt
## Pretrained Model
- [Google Drive](https://drive.google.com/file/d/19oQCPPe5w1vUDzdfD5OXyDvmbE9QLGv8/view?usp=sharing)
## Tips
- For the training process, it is recommended to have a batch size equal to or larger than 4. Having a batch size smaller than 4 may lead to training instabilities and convergence issues.
