import os
import argparse
import sys

tipo = sys.argv[1]

if tipo == "1":
    os.system("""python extract_embeddings.py --dataset dataset \
    --embeddings output/embeddings.pickle \
    --detector face_detection_model \
    --embedding-model openface_nn4.small2.v1.t7""")
    os.system("""python train_model.py --embeddings output/embeddings.pickle \
    --recognizer output/recognizer.pickle \
    --le output/le.pickle""")
elif tipo == "2":
    chamada = chamada = sys.argv[2]
    os.system("""python recognize_video.py --chamada {} \
    --detector face_detection_model \
    --embedding-model openface_nn4.small2.v1.t7 \
    --recognizer output/recognizer.pickle \
    --le output/le.pickle""".format(chamada))
else:
  print("Tipo desconhecido")
