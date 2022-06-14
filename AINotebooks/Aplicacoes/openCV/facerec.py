# -*- coding: utf-8 -*-
"""FaceRec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1h5D30M6Hl5HQl0hdQq3c-ARAbMCIwwCU

Instalando o OpenCV

Primeiro, você precisa encontrar o arquivo de configuração correto para o seu sistema operacional.

Descobri que instalar o OpenCV foi a parte mais difícil da tarefa. Se você receber erros estranhos e inexplicáveis, pode ser devido a conflitos de biblioteca, diferenças de 32/64 bits e assim por diante. Achei mais fácil usar uma máquina virtual Linux e instalar o OpenCV do zero.

Depois de concluir a instalação, você pode testar se funciona ou não, iniciando uma sessão do Python e digitando:
"""

import cv2

"""Se você não receber nenhum erro, poderá passar para a próxima parte.
Entendendo o Código

Vamos detalhar o código real, que você pode baixar do repositório. Pegue o script face_detect.py, a foto abba.png e o haarcascade_frontalface_default.xml.
"""

# Abrir uma imagem e uma cascata pronta
imagePath = "galera-reunida.jpg"
cascPath = "haarcascades/haarcascade_frontalface_default.xml"

"""Você primeiro passa os nomes da imagem e da cascata como argumentos de linha de comando. Usaremos a imagem do ABBA, bem como a cascata padrão para detecção de rostos fornecida pelo OpenCV."""

# Criar o classificador em cascata via xml suplementado
faceCascade = cv2.CascadeClassifier(cascPath)

"""Agora criamos a cascata e inicializamos com nossa cascata de rosto. Isso carrega a cascata de rostos na memória para que esteja pronta para uso. Lembre-se, a cascata é apenas um arquivo XML que contém os dados para detectar rostos."""

# Ler a imagem
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

"""Aqui lemos a imagem e a convertemos em escala de cinza. Muitas operações no OpenCV são feitas em escala de cinza.

"""

# Detectar os objetos na imagem
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(10, 10),
    flags = cv2.CASCADE_SCALE_IMAGE
)

"""Esta função detecta o rosto real e é a parte principal do nosso código, então vamos ver as opções:

A função detectMultiScale é uma função geral que detecta objetos. Como estamos chamando na cascata de rosto, é isso que ele detecta.

A primeira opção é a imagem em tons de cinza.

O segundo é o fator de escala. Como alguns rostos podem estar mais próximos da câmera, eles parecem maiores do que os rostos na parte de trás. O fator de escala compensa isso.

O algoritmo de detecção usa uma janela móvel para detectar objetos. minNeighbors define quantos objetos são detectados próximos ao atual antes de declarar o rosto encontrado. minSize, enquanto isso, fornece o tamanho de cada janela.

Nota: Eu peguei valores comumente usados ​​para esses campos. Na vida real, você experimentaria valores diferentes para o tamanho da janela, fator de escala e assim por diante, até encontrar um que funcione melhor para você.

A função retorna uma lista de retângulos nos quais acredita ter encontrado um rosto. Em seguida, faremos um loop sobre onde ele acha que encontrou algo.
"""

print ("Encontrados", len(faces),"rostos!")

# Desenha um retângulo ao redor dos rostos
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

"""Esta função retorna 4 valores: a localização x e y do retângulo e a largura e altura do retângulo (w , h).

Usamos esses valores para desenhar um retângulo usando a função retângulo() embutida.
"""

cv2.imshow("Rostos encontrados", image)
cv2.waitKey(0)

"""No final, exibimos a imagem e esperamos que o usuário pressione uma tecla."""

cascPath2 = "haarcascades/haarcascade_smile.xml"
# Criar o classificador em cascata via xml suplementado
smileCascade = cv2.CascadeClassifier(cascPath2)

# Detectar os objetos novamente na imagem
for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]
    smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=2)
                                           
    # Desenha um novo retângulo ao redor dos sorrisos
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 255), 2)

cv2.destroyAllWindows()
cv2.imshow("Sorrisos encontrados", image)
cv2.waitKey(0)
cv2.destroyAllWindows()