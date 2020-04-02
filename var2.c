#include "mpi.h"

#include "unistd.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#define _USE_MATH_DEFINES
#include "math.h"
#include "time.h"
#include "MatrixOperations.h"

#define EPS 0.0000001
#ifndef M_PI
#define M_PI 3.1415
#endif
int size, rank;

struct Part {
    int id;
    int size;
};

void share(double *, int);
struct Part getPart(int i, int N);
void printMat(double *mat, int I, int J);
void sumColumns (double *vector, const int N);

void buildCoefficientMatrix(double *matrix, int N, int partSize, int partId) {
    int tag = 12432;

    if (rank == 0) {
        int i, j, k, l;
        double *A = (double *) malloc(sizeof(double) * N * N);
        if (A == NULL) {
            perror("malloc\n");
            exit(1);
        }

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i * N + j] = (double)rand() / (RAND_MAX / 2);
            }
        }

        for (k = 0; k < size; k++) {
            struct Part part = getPart(k, N);
            double *buff = NULL;

            if (k == rank) {
                buff = matrix;
            } else {
                buff = (double *) malloc(sizeof(double) * N * part.size);
                if (buff == NULL) {
                    perror("malloc\n");
                    exit(1);
                }
            }

            for (i = 0; i < N; i++) {
                for (j = 0; j < part.size; j++) {
                    buff[i * part.size + j] = 0;
                    for (l = 0; l < N; l++) {
                        buff[i * part.size + j] += A[N * l + i] * A[N * l + j + part.id];
                    }
                }
            }
            if (k != rank) {
                MPI_Send(buff, N * part.size, MPI_DOUBLE, k, tag, MPI_COMM_WORLD);
                free(buff);
            }
        }
        free(A);
    } else {
        MPI_Recv(matrix, N * partSize, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void buildRightVector(double *vector, const double *coefficientMatrix, int N, int partSize, int partId) {
    double *u = (double *)malloc(sizeof(double) * N);
    int i;
    for (i = 0; i < N; i++) {
        u[i] = sin((2 * M_PI * i) / N);
    }

    dot_mv(coefficientMatrix, u + partId, N, partSize, vector);
    sumColumns(vector, N);
    free(u);
}

void printMat(double *mat, int I, int J) {
    int i, j;
    for (i = 0; i < I; i++) {
        for (j = 0; j < J; j++) {
            printf("%.40lf ", mat[i * J + j]);
        }
        printf("\n");
    }
    printf("\n");
}

struct Part getPart(int i, int N) {
    int partSize = N / size;
    int id = partSize * i;
    if (i == size - 1) {
        partSize = N - id;
    }
    struct Part out = {id, partSize};
    return out;
}

void share(double *result, int I) {
    int tag = 1;
    for (int i = 0; i < size - 1; i++) {
        struct Part sendPart = getPart((rank - i + size) % size, I);
        struct Part recvPart = getPart((rank - i - 1 + size) % size, I);
        MPI_Sendrecv(result + sendPart.id, sendPart.size, MPI_DOUBLE, (rank + 1) % size, tag,
                     result + recvPart.id, recvPart.size, MPI_DOUBLE, (rank - 1 + size) % size, tag, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
    }
}

void sumColumns (double *vector, const int N) {
    /*int tag = 1;
    int i, j;
    double *sendBuffer = (double *)malloc(sizeof(double) * N);
    double *recvBuffer = (double *)malloc(sizeof(double) * N);
    double *b;

    if (sendBuffer == NULL || recvBuffer == NULL) {
        printf("malloc\n");
        exit(1);
    }
    memcpy(sendBuffer, vector, sizeof(double) * N);
    for (j = 0; j < size - 1; j++) {
        MPI_Sendrecv(sendBuffer, N, MPI_DOUBLE, (rank + 1) % size, tag,
                     recvBuffer, N, MPI_DOUBLE, (rank - 1 + size) % size, tag, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        for (i = 0; i < N; i++) {
            vector[i] += recvBuffer[i];
        }
        b = sendBuffer;
        sendBuffer = recvBuffer;
        recvBuffer = b;
    }
    free(sendBuffer);
    free(recvBuffer);*/
    double *buff = (double *)malloc(sizeof(double) * N);
    MPI_Allreduce(vector, buff, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(vector, buff, sizeof(double) * N);
    free(buff);
}

int main(int argc, char **argv) {
    double *A;
    double *b;
    double *x;
    double *r;
    double *z;
    double *A_mul_z;
    double alpha, beta;
    double dot_rr, norm_b, Az_dot_z, dot_rr_new;
    double timer;
    int N, i;
    struct Part part;
    double deviation;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        N = 10;
    } else {
        N = atoi(argv[1]);
    }

    part = getPart(rank, N);

    A = (double *) malloc(sizeof(double) * N * part.size);
    b = (double *) malloc(sizeof(double) * N);
    x = (double *) malloc(sizeof(double) * N);
    r = (double *) malloc(sizeof(double) * N);
    z = (double *) malloc(sizeof(double) * N);
    A_mul_z = (double *) malloc(sizeof(double) * N);

    if (A == NULL || b == NULL || x == NULL || r == NULL || z == NULL || A_mul_z == NULL) {
        perror("malloc\n");
        exit(1);
    }
    printf("Memory allocated in process %d\n", rank);

    buildCoefficientMatrix(A, N, part.size, part.id);

    buildRightVector(b, A, N, part.size, part.id);

    for (i = 0; i < N; i++) {
        x[i] = 0;
        r[i] = z[i] = b[i];
    }



    printf("Time measuring started in process %d\n", rank);
    timer = MPI_Wtime();

    norm_b = dot_vv(b, b, N);
    norm_b = sqrt(norm_b);
    dot_rr = dot_vv(r, r, N);



    int da = 0;
    while (1) {
        dot_mv(A, z + part.id, N, part.size, A_mul_z);
        sumColumns(A_mul_z, N);

        Az_dot_z = dot_vv(A_mul_z, z, N);

        alpha = dot_rr / Az_dot_z;

        add_v_mul_sv(z, alpha, x, N, x);

        add_v_mul_sv(A_mul_z, -alpha, r, N, r);

        dot_rr_new = dot_vv(r, r, N);

        beta = dot_rr_new / dot_rr;

        add_v_mul_sv(z, beta, r, N, z);
        if ((deviation = (sqrt(dot_rr) / norm_b)) <= EPS) {
            break;
        }
        dot_rr = dot_rr_new;
        da++;
    }

    timer = MPI_Wtime() - timer;

    printf("Process %d time: %lf\n", rank, timer);
    dot_mv(A, x + part.id, N, part.size, A_mul_z);
    sumColumns(A_mul_z, N);
    if (rank == 0) {
        double result;
        add_v_mul_sv(A_mul_z, -1, b, N, A_mul_z);
        result = dot_vv(A_mul_z, A_mul_z, N);
        printf("result: %lf\n %d iterations\n", result, da);
    }
    free(A);
    free(b);
    free(x);
    free(r);
    free(z);
    free(A_mul_z);

    MPI_Finalize();

    return 0;
}

