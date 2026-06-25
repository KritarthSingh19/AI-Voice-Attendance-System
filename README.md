# AI Voice Attendance System

Current Stage: Speaker Recognition Prototype

## Overview

This project started as a personal learning project to understand how speaker recognition systems work using Machine Learning and Digital Signal Processing.

Instead of following tutorials, I wanted to build something practical that combines AI with a real-world problem.

The long-term goal is to build a secure multimodal attendance system that combines voice recognition and face recognition to verify a user's identity before attendance is recorded.

So far, I have built the speaker recognition component of the system. The current pipeline can:

- Record voice samples
- Extract audio features
- Train a speaker recognition model
- Identify enrolled speakers from voice recordings

The project is being developed as part of my learning journey in Artificial Intelligence, Machine Learning, Computer Vision, and Data Science.

## Current Features

### Voice Enrollment
Record and store voice samples for speaker enrollment.

### Audio Feature Extraction
Extract MFCC (Mel-Frequency Cepstral Coefficients) features from recorded voice samples.

### Speaker Recognition
Train a machine learning model capable of identifying enrolled speakers based on their voice.

### Speaker Prediction
Predict the identity of a speaker from a given voice recording.

### Model Persistence
Save trained models and label encoders for future use without retraining.


## Tech Stack

### Programming Language

- Python

### Libraries and Tools

- NumPy
- Pandas
- Librosa
- Scikit-Learn
- SoundDevice
- SoundFile
- Joblib
- Git
- GitHub

### Machine Learning Concepts

- Audio Signal Processing
- MFCC Feature Extraction
- Speaker Recognition
- Random Forest Classification

## What I Learned

While building this project, I learned:

- How audio signals are represented and processed digitally
- How MFCC features are extracted from speech recordings
- How machine learning models can classify speakers
- How to build a complete data processing pipeline
- How to organize an AI project from data collection to prediction
- How to use Git and GitHub for version control
- The challenges involved in biometric authentication systems


## Installation

Clone the repository:

```bash
git clone https://github.com/KritarthSingh19/AI-Voice-Attendance-System.git

cd AI-Voice-Attendance-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Record Voice Samples

```bash
python scripts/audio/record_voice.py
```

### Extract Features

```bash
python scripts/features/extract_features.py
```

### Train Model

```bash
python scripts/models/train_model.py
```

### Predict Speaker

```bash
python scripts/models/predict_speaker.py
```

## Current Architecture

```text
Voice Input
     │
     ▼
Audio Recording
     │
     ▼
MFCC Feature Extraction
     │
     ▼
Random Forest Classifier
     │
     ▼
Speaker Prediction
```

## Current Limitations

This project is still in the prototype stage.

Current limitations include:

- Small training dataset
- Limited number of enrolled speakers
- No anti-spoof detection
- Attendance logging not implemented
- No web dashboard
- Face recognition module not yet integrated

These limitations will be addressed in future versions.

## Development Roadmap

### Completed

- Voice recording module
- Voice enrollment pipeline
- MFCC feature extraction
- Speaker recognition model
- Speaker prediction pipeline
- Model saving and loading

### In Progress

- Face recognition module

### Planned

- Voice and Face verification
- Attendance marking system
- Attendance database
- Attendance analytics dashboard
- Anti-spoof detection
- Real-time biometric authentication
- Web-based user interface

## Why I Built This

Most attendance systems rely on manual entry, RFID cards, fingerprints, or QR codes.

I wanted to explore how Machine Learning and Biometrics can be combined to create a smarter and more secure attendance system.

As the project evolves, I plan to integrate face recognition, attendance logging, anti-spoof detection, and analytics to create a complete biometric attendance platform.

This project allows me to gain practical experience in:

- Machine Learning
- Audio Processing
- Computer Vision
- Biometric Authentication
- Software Development
- End-to-End AI Project Development

## About Me

I'm Kritarth Singh, a B.Tech Computer Science student with a strong interest in:

- Artificial Intelligence
- Machine Learning
- Data Science
- Analytics

I enjoy building hands-on projects that help me understand concepts beyond theory and apply them to real-world problems.

My goal is to build practical AI-driven solutions, strengthen my software engineering skills, and gain hands-on experience through real-world projects.