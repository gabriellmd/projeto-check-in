import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required=True,
    help="Você deve informar um tipo de execução. 1 para extrair e treinar. 2 para abrir o reconhecedor em video")
ap.add_argument("-ch", "--chamada", required=True,
	help="path to csv file to write presence")
args = vars(ap.parse_args())

tipo = args["type"]
chamada = args["chamada"]
print(chamada)
print(tipo)

if tipo == "1":
  os.system("""python extract_embeddings.py --dataset dataset \
  --embeddings output/embeddings.pickle \
  --detector face_detection_model \
  --embedding-model openface_nn4.small2.v1.t7""")
  os.system("""python train_model.py --embeddings output/embeddings.pickle \
  --recognizer output/recognizer.pickle \
  --le output/le.pickle""")
elif tipo == "2":
  os.system("""python recognize_video.py --chamada {} \
  --detector face_detection_model \
  --embedding-model openface_nn4.small2.v1.t7 \
  --recognizer output/recognizer.pickle \
  --le output/le.pickle""".format(chamada))
else:
  print("Tipo desconhecido")
