import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ising_numba import run_simulation

def validate_args(args):
    if args.beta < 0:
        raise ValueError(f"Parametr beta nie może być ujemny. Podano: {args.beta}")
    if args.N <= 0:
        raise ValueError(f"Rozmiar siatki (N) musi być dodatni. Podano: {args.N}")
    if args.M <= 0:
        raise ValueError(f"Liczba makrokroków (M) musi być dodatnia. Podano: {args.M}")

def main():
    parser = argparse.ArgumentParser(
        description="Symulacja modelu Isinga.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--N", type=int, default=100, help="Rozmiar siatki NxN.")
    parser.add_argument("--M", type=int, default=500, help="Liczba makrokroków.")
    parser.add_argument("--beta", type=float, default=0.4, help="Parametr beta.")
    parser.add_argument("--B", type=float, default=0.0, help="Pole zewnętrzne B.")
    parser.add_argument("--J", type=float, default=1.0, help="Stała oddziaływania J.")
    
    parser.add_argument("--magnetization-file", type=str, default=None, help="Nazwa pliku do zapisu magnetyzacji.")
    parser.add_argument("--show-animation", action="store_true", help="Wyświetla okno z animacją po zakończeniu obliczeń.")
    parser.add_argument("--animation-file", type=str, default=None, help="Nazwa pliku do zapisu animacji.")

    try:
        args = parser.parse_args()
        validate_args(args)
    except ValueError as e:
        print(f"\nBŁĄD DANYCH WEJŚCIOWYCH: {e}")
        sys.exit(1)
    except SystemExit:
        sys.exit(0)

    print(f"\nUruchamianie symulacji: N={args.N}, M={args.M}, beta={args.beta}, J={args.J}, B={args.B}")
    start_time = time.perf_counter()
    
    try:
        grid_history, energy_history, mag_history = run_simulation(args.N, args.J, args.B, args.beta, args.M)
    except RuntimeError as e:
        print(f"\nBŁĄD SILNIKA SYMULACJI: {e}")
        sys.exit(1)
        
    end_time = time.perf_counter()
    print(f"Symulacja zakończona. Czas wykonania: {end_time - start_time:.4f} s.")

    if args.magnetization_file:
        try:
            steps = np.arange(args.M)
            data_to_save = np.column_stack((steps, mag_history))
            ext = args.magnetization_file.split('.')[-1].lower()
            sep = "," if ext == "csv" else " "
            
            np.savetxt(args.magnetization_file, data_to_save, 
                       fmt=['%d', '%.6f'], delimiter=sep, 
                       header="Step Magnetization", comments='')
            print(f"Historię magnetyzacji zapisano do: {args.magnetization_file}")
            
        except OSError as e:
            print(f"\nBŁĄD ZAPISU DANYCH: Odmowa dostępu lub błąd ścieżki: {e}")

    if args.show_animation or args.animation_file:
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title(f"Model Isinga (beta={args.beta})")
            ax.axis('off')
            
            img = ax.imshow(grid_history[0], vmin=-1, vmax=1, cmap='coolwarm')
            
            def update(frame):
                img.set_array(grid_history[frame])
                return img,
                
            anim = animation.FuncAnimation(fig, update, frames=args.M, interval=50, blit=False)
            
            if args.animation_file:
                ext = args.animation_file.split('.')[-1].lower()
                if ext not in ['gif', 'mp4']:
                    raise ValueError(f"Format '.{ext}' nie jest wspierany.")
                
                print(f"Przygotowywanie animacji...")
                if ext == "gif":
                    anim.save(args.animation_file, writer='pillow', fps=20)
                elif ext == "mp4":
                    anim.save(args.animation_file, writer='ffmpeg', fps=20)
                    
                print(f"Animację zaspisano do: {args.animation_file}")
                
            if args.show_animation:
                print("Otwieranie okna z animacją...")
                plt.show()
                
        except ValueError as e:
            print(f"\nBŁĄD ANIMACJI: {e}")
        except OSError as e:
            print(f"\nBŁĄD ZAPISU ANIMACJI: Problem z systemem plików: {e}")

if __name__ == "__main__":
    try:
        main()
        
    except KeyboardInterrupt:
        print("\n\nSYMULACJA ZATRZYMANA.")
        sys.exit(130)
        
    except Exception as e:
        print(f"\nKRYTYCZNY BŁĄD NIEZNANEGO TYPU.")
        print(f"Szczegóły błędu: {e}")
        sys.exit(1)