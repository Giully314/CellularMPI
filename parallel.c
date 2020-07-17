#include <mpi/mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define DEBUG(w, i) printf("%d: %s\n", i, w)


#define err(s) { perror((s)); exit(EXIT_FAILURE); }
#define MALLOC(s,t) if(((s) = malloc(t)) == NULL) { err("error: malloc() "); }

#define CART_DIMENSIONS 1

#define ROOT 0

#define ROWS_PER_PROC 5
#define SIZE_COLUMNS 70
#define GHOST_SIZE 2

//Dichiaro le variabili globali per semplificare il programma nella scrittura e renderlo più leggibile.
//In una versione ufficiale, utilizzerei un altro approccio ovviamente, definendo un pattern comune a tutte le funzioni.

//Rank dei processi vicini.
int neighboors[2];

MPI_Request requests[4];
MPI_Status status[4];

MPI_Comm cart_comm;



// Fine variabili globali



//Creazione di un array 2d contiguo in memoria.
int** Make2DArray(int rows, int columns);

void Free2DArray(int **a);


/*
Creazione popolazione cellular automata. Questa funzione può essere personalizzata per creare varie
popolazioni di automi cellulari. 
*/
void PopulateCellularAutomata(int** cellular, int rows, int columns);

//Conta le cellule vive intorno alla cellula nella posizione i,j.
int CountNeighbors(int** a, int i, int j);


//avanza di step.
void NextStep(int rank, int** a, int** update);


//Scambia i ghost points
void Exchange(int** a);

//Questa funzione può essere modificata per implementare qualsiasi regola per gli automi cellulari.
void CheckRule(int count, int** a, int i, int j, int** update);


//int GetCartRank(MPI_Comm cart_comm);
void FindNeighbors(MPI_Comm cart_comm, int rank, int *neighbors);


void Print2DArray(int *a, int rows, int columns)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            if (a[i * columns + j] == 1)
            {
                printf("\033[0;32m"); //setto il colore
                printf("# ");
            }
            else
            {
                printf("\033[0m");//resetto il colore
                printf("0 ");
            }
        }
        printf("\n");
    }

    printf("\n\n");
}


int main(int argc, char** argv)
{
    //Inizializazzione risorse di base MPI ed info rank.
    MPI_Init(&argc, &argv);

    int rank;
    int num_of_tasks;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_tasks);


    //Setup creazione topologia cartesiana ad una dimensione.

    int dims[CART_DIMENSIONS] = { 0 };
    int periodic[CART_DIMENSIONS] = { 1 }; //periodico.


    //Per dimensioni >= 2, fare il check se tutti i processi sono in uso.
    MPI_Dims_create(num_of_tasks, CART_DIMENSIONS, dims);

    //Momentaneamente non permetto di riordinare i rank.
    MPI_Cart_create(MPI_COMM_WORLD, CART_DIMENSIONS, dims, periodic, 0, &cart_comm);

    if (cart_comm == MPI_COMM_NULL)
    {
        printf("Error cart");
    }

    FindNeighbors(cart_comm, rank, neighboors);

    
    //printf("RANK %d:  %d %d\n", rank, neighboors[0], neighboors[1]);
    


    int* cellular_popolation = NULL;
    
    /*
    Si può anche utilizzare un'altra strategia piuttosto che inizializzare i dati in un processo root e poi usare scatter.
    Si può implementare un sistema che permetta ai processi di inizializzare in automatico la loro parte così da evitare
    un overhead iniziale tra inizializzazione dei dati da parte di un solo processo e scatter.
    */

    const int rows = num_of_tasks * ROWS_PER_PROC;
    if (rank == ROOT)
    {
        //cellular_popolation = Make2DArray(rows, SIZE_COLUMNS);

        /*
        Devo allocare in modo "diverso" l'array principale poiché scatter non riconosce che la memoria è contigua.
        */
        MALLOC(cellular_popolation, rows * SIZE_COLUMNS * sizeof(int));

        srand(13);
        for (int i = 0; i < rows * SIZE_COLUMNS; ++i)
        {
            cellular_popolation[i] = 0;
        }

        int i ;
        for (i = 20; i < 20 + 8; ++i)
        {
            cellular_popolation[20 * SIZE_COLUMNS + i] = 1;
        }
        ++i;

        int j = i;
        for (; i < j + 5; ++i)
        {
            cellular_popolation[20 * SIZE_COLUMNS + i] = 1;
        }

        i += 3;
        j = i;
        for (; i < j + 3; ++i)
        {
            cellular_popolation[20 * SIZE_COLUMNS + i] = 1;
        }

        i += 6;
        j = i;
        for (; i < j + 7; ++i)
        {
            cellular_popolation[20 * SIZE_COLUMNS + i] = 1;
        }


        i += 1;
        j = i;
        for (; i < j + 5; ++i)
        {
            cellular_popolation[20 * SIZE_COLUMNS + i] = 1;
        }


        /* #define cell(a, x, y) a[y * SIZE_COLUMNS + x]

        cell(cellular_popolation, 41, 40) = 1;
        cell(cellular_popolation, 42, 40) = 1;

        cell(cellular_popolation, 40, 41) = 1;
        cell(cellular_popolation, 41, 41) = 1;

        cell(cellular_popolation, 40, 42) = 1; */
        

        /* PopulateCellularAutomata(cellular_popolation, rows, SIZE_COLUMNS);
        */
    }


    //int **cellular_group = Make2DArray(ROWS_PER_PROC, SIZE_COLUMNS);


    const int data_length = ROWS_PER_PROC * SIZE_COLUMNS;
    int** cellular_group = Make2DArray(ROWS_PER_PROC + GHOST_SIZE, SIZE_COLUMNS);
    
    //int *data = cellular_popolation[0];

    MPI_Scatter(cellular_popolation, data_length, MPI_INT, &(cellular_group[1][0]), data_length, MPI_INT, ROOT,
                    cart_comm); 
    



    int **updated_cellular = Make2DArray(ROWS_PER_PROC, SIZE_COLUMNS);
    memset(&updated_cellular[0][0], 0, ROWS_PER_PROC * SIZE_COLUMNS * sizeof(int));

    int number_of_cycles = 10;

    while (number_of_cycles >= 0)
    {
        /*
        In una possibile versione ottimizzata dopo aver provato il programma su un sistema distribuito, si potrebbe
        scrivere l'operazione di scambio dei dati in modo più perfomante. Si potrebbe iniziare un'operazione di send,
        computare i dati, e poi riceverli, ripetendo così il ciclo. IN questo modo si riducono al minimo i tempi 
        di latenza della memoria, che in un sistema distribuito, sono i più importanti da risolvere con la scrittura
        di codice che ne tenga conto. Anche per quanto riguarda l'ouput delle informazioni vale questa cosa.
        */
        NextStep(rank, cellular_group, updated_cellular);

        MPI_Gather(&updated_cellular[0][0], ROWS_PER_PROC * SIZE_COLUMNS, MPI_INT, cellular_popolation, 
                    ROWS_PER_PROC * SIZE_COLUMNS, MPI_INT, ROOT, cart_comm); 

        
        if (rank == ROOT)
        {
            Print2DArray(cellular_popolation, rows, SIZE_COLUMNS);
        }


        sleep(1);

        --number_of_cycles;
    }




    Free2DArray(cellular_group);
    Free2DArray(updated_cellular);
    
    if (rank == ROOT)
    {
        free(cellular_popolation);
    }

    MPI_Finalize();

    return 0;
}




//************************************+ DEIFINIZIONE FUNZIONI *************************************************
int** Make2DArray(int rows, int columns)
{
    int *data = malloc(rows * columns * sizeof(int));

    //printf("%p\n", data);


    int **array = malloc(rows * sizeof(int*));

    if (array == NULL || data == NULL)
    {
        printf("error\n");
    }


    for (int i = 0; i < rows; ++i)
    {
        array[i] = &(data[i * columns]);
    }


    return array;
}


void Free2DArray(int** a)
{
    free(a[0]);
    free(a);
}


void FindNeighbors(MPI_Comm cart_comm, int rank, int *neighbors)
{
    int coords[1];

    MPI_Cart_coords(cart_comm, rank, CART_DIMENSIONS, coords);

    int i = coords[0];
    coords[0] = i - 1;
    MPI_Cart_rank(cart_comm, coords, &neighbors[0]);

    coords[0] = i + 1;
    MPI_Cart_rank(cart_comm, coords, &neighbors[1]);
}



void PopulateCellularAutomata(int** cellular, int rows, int columns)
{
    //parametro momentaneo!
    //srand(0);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            cellular[i][j] = 42;//rand() % 2;
        }
    }
}


int CountNeighbors(int** a, int i, int j)
{
    //printf("%d %d\n",a[0][0], a[1][0]);
    int count = a[i - 1][j - 1] + a[i - 1][j] + a[i - 1][j + 1] +
                a[i][j - 1]     +       0     + a[i][j + 1]     +
                a[i + 1][j - 1] + a[i + 1][j] + a[i + 1][j + 1];

    return count;
}


void NextStep(int rank, int** a, int** update)
{

    /* printf("rank %d   neighboors: %d %d\n", rank, neighboors[0], neighboors[1]);
    printf("Before. rank: %d. firsRow: %d %d %d %d %d\n", rank, a[0][0], a[0][1], a[0][2], a[0][3], a[0][4]);
    printf("Before. rank: %d. lastRow: %d %d %d %d %d\n", rank, a[4][0], a[4][1], a[4][2], a[4][3], a[4][4]); */
    
    /* char s[SIZE_COLUMNS + 1];
    for (int i= 0; i < SIZE_COLUMNS; ++i)
    {
        sprintf(&s[i], "%d", a[0][i]);
    }
    s[SIZE_COLUMNS] = '\0';
    printf("Before. rank %d  %s\n", rank, s); */

    Exchange(a);

    /* for (int i= 0; i < SIZE_COLUMNS; ++i)
    {
        sprintf(&s[i], "%d", a[0][i]);
    } */
    /* s[SIZE_COLUMNS] = '\0';
    printf("After . rank %d  %s\n\n", rank, s); */
    

    /* printf("After.  rank: %d. firsRow: %d %d %d %d %d\n", rank, a[0][0], a[0][1], a[0][2], a[0][3], a[0][4]);
    printf("After.  rank: %d. lastRow: %d %d %d %d %d\n\n", rank, a[4][0], a[4][1], a[4][2], a[4][3], a[4][4]); */


    for (int i = 1; i < ROWS_PER_PROC + 1; ++i)
    {
        for (int j = 1; j < SIZE_COLUMNS - 1; ++j)
        {
            int count = CountNeighbors(a, i, j);
           
            CheckRule(count, a, i, j, update);
            memcpy(&a[1][0], &update[0][0], ROWS_PER_PROC * SIZE_COLUMNS * sizeof(int));
        }
    }
}


void Exchange(int** a)
{
    MPI_Isend(&a[3][0], SIZE_COLUMNS, MPI_INT, neighboors[1], 0, cart_comm, &requests[0]);
  
    MPI_Isend(&a[1][0], SIZE_COLUMNS, MPI_INT, neighboors[0], 0, cart_comm, &requests[1]);

    MPI_Irecv(&a[0][0], SIZE_COLUMNS, MPI_INT, neighboors[1], 0, cart_comm, &requests[2]);

    MPI_Irecv(&a[3][0], SIZE_COLUMNS, MPI_INT, neighboors[0], 0, cart_comm, &requests[3]);


    MPI_Waitall(4, requests, status);
}


void CheckRule(int count, int **a, int r, int c, int **update)
{

    if (count < 2 || count > 3)
    {
        update[r-1][c] = 0;
    }
    else if ((count == 2 || count == 3) && a[r][c] == 1)
    {
        update[r-1][c] = 1;
    }
    else if (count == 3)
    {
        update[r-1][c] = 1;
    }
}