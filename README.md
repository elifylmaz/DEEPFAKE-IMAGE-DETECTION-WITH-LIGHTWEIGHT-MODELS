# Deepfake Image Detection with Lightweight Models

## 📄 Overview

This project focuses on developing lightweight and efficient deep learning models for detecting deepfake images, addressing the growing threat of media manipulation. By leveraging Convolutional Neural Networks (CNNs) and advanced face detection methods, this study provides high accuracy while maintaining low computational cost, making it practical for real-world applications in security and defense.

## 📝 Abstract

Deepfake technologies pose serious challenges to information security and public trust, enabling highly realistic synthetic content creation. In this study, four different approaches were explored using the Celeb-DF v2 dataset (345,066 training images and 86,482 test images):

1. **Direct frame-based training**
2. **MTCNN-based face extraction with knowledge distillation and pruning**
3. **OpenCV DNN-based face extraction**
4. **Direct use of MTCNN-extracted face images**

Models such as ResNet18, ResNet50, MobileNetV2, MobileNetV3, EfficientNet-B0, and MesoNet were utilized and trained on an NVIDIA GeForce RTX 4060 GPU. The highest performance (97% accuracy) was achieved by EfficientNet-B0 using MTCNN-based face extraction.

## ⚙️ Methods

### Dataset

- **Celeb-DF v2**: Includes 5,639 fake and 890 real videos.
- Frame extraction: 30 frames per video using OpenCV.
- Augmentation techniques: Horizontal flip, brightness/contrast adjustment, rotation, Gaussian blur, and color jitter (implemented with the Albumentations library).

### Models and Techniques

- **CNN Architectures**: ResNet18, ResNet50, MobileNetV2, MobileNetV3, EfficientNet-B0, and MesoNet.
- **Face Detection**: MTCNN and OpenCV DNN.
- **Lightweight Optimization**: Knowledge distillation and pruning techniques to reduce computational load while preserving accuracy.

## ✅ Results

| Method                                | Model(s)                     | Accuracy (%) |
|---------------------------------------|-----------------------------|--------------|
| Direct frame-based training           | EfficientNet-B0, MobileNetV2 | 96           |
| Knowledge distillation & pruning      | ResNet50 (teacher), MesoNet (student) | 77–80 |
| OpenCV DNN face extraction           | EfficientNet-B0, MobileNetV3, ResNet18, MesoNet | 66–82 |
| MTCNN face extraction                | EfficientNet-B0, MobileNetV2, ResNet18 | **97** |

The highest performance was achieved using MTCNN face extraction with EfficientNet-B0.

## 💡 Applications

- Defense and security
- Media forensics
- Social media content verification

## ⭐ Highlights

- Advanced face detection significantly improves accuracy by focusing on facial regions and reducing background noise.
- Lightweight models enable real-time deepfake detection on low-resource devices.
- Knowledge distillation and pruning effectively reduce computational costs while maintaining model performance.

## 👨‍💻 Authors

- Ardıl Silan Aydın
- Neslihan Özdil
- Elif Yılmaz

**Supervisor**: Asst. Prof. Dr. Samsun Mustafa Başarıcı

## 🙏 Acknowledgements

This work was supported within the ATP 2024–2025 framework, in collaboration with the Presidency of Defense Industries (SSB) and the Council of Higher Education (YÖK), coordinated by SAYZEK. Special thanks to our mentor Emrah Başaran for his technical guidance and continuous support.

## 📚 References

- Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics", CVPR, 2020.
- Hinton et al., "Distilling the Knowledge in a Neural Network", arXiv preprint, 2015.
- Zhang et al., "Joint Face Detection and Alignment Using Multi-task Cascaded Convolutional Networks", IEEE SPL, 2016.
- Buslaev et al., "Albumentations: Fast and Flexible Image Augmentations", Information, 2020.
- OpenCV Documentation.

## 📝 License

This repository is open for academic and research purposes. Please cite the authors when using or referring to this work.

---

