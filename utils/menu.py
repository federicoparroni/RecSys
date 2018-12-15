import utils.log as log
import random

random_byes = ['Ciao stupido!', 'Baci dalla mamma dell\'Alan Fuck <3', 'Morzenti dice di no',
                'Wait until the 80s and...', 'Do you want a candy? >O<', 'Vaffanculo agnicosa',
                'Vedo la tu mamma in posizioni mamma mia', 'E\' stato meglio di una serata con gli architetti',
                'Che ne dici di organizzare una cèna?', 'Grazie Edo', 'I used the venishing gridient']

def quit_menu(main_menu):
    if main_menu:
        print()
        print(random.choice(random_byes))

def show(title, options, can_exit=True, main_menu=False, decorator='++'):
    """
    Display a menu

    options: dictionary in which each key is a string and each value is a tuple (string, function), representing
            the text of the function that will be called when the related string in inserted as input
        ex: { 'a', ('option a', print) } : print 'option a' and when 'a' is pressed, call the function 'print'
    """
    log.success('{} {} {}'.format(decorator, title, decorator))
    for s,f in options.items():
        log.warning('({}) {}'.format(s,f[0]))
    if can_exit:
        log.warning('(x) Exit')
    
    wrong_choice = True
    while(wrong_choice):
        arg = input()
        print()

        try:
            if arg=='x' and can_exit:
                wrong_choice = False
                quit_menu(main_menu)
            else:
                funct = options[arg][1]
                wrong_choice = False
                res = funct()
                quit_menu(main_menu)
                return res
        except KeyError as _:
            log.error('Invalid option, retry:')

# if __name__ == "__main__":
#     show('prova', { 'v': ('option 1', print) } )
