import os

def menu():
    funcoes = ['Atualizar base de dados', 'Sair']
    if os.path.isfile('trainer/trainer.yml'):
        funcoes.insert(1, 'Abrir Reconhecedor')

    menu = ''
    for i in range(len(funcoes)):
        menu += '{}. {}\n'.format(i+1, funcoes[i])
    print(menu)

while True:
    os.system('clear')
    menu()
    op = int(input("=> "))
    if op == 1:
        name = input("Digite a matricula da pessoa => ")
        os.system("python save_face.py '{}'".format(name))
        os.system("python face_training.py")
    elif op == 2:
        os.system("python reconhecedor.py")
    else:
        break
