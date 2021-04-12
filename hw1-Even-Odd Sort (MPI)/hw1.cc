#include <cstdio>
#include <mpi.h>
#include <stdlib.h>
#include <cstring>
#include <boost/sort/spreadsort/spreadsort.hpp>

int main(int argc, char** argv) {

	MPI_Init(&argc,&argv);
	int rank, size;
	unsigned int data_length, read_pos, rest;
	unsigned int num = (atoll(argv[1])); // the size of the array
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Request request, request2;

	data_length = num/(size);
	rest = num%(size);

	unsigned int recv_data_length = data_length;

	if(rank+1 < size && rank+1 < rest) ++recv_data_length;
	if( rank < rest ) {
		++data_length;
		read_pos = data_length*rank;
	}else{
		read_pos = rest + rank*data_length;
	}

	MPI_File f;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
	float *data = (float*) malloc(sizeof(float) * (data_length << 1));
	
	if(rank<num) {
        MPI_File_read_at(f, sizeof(float) * read_pos, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);
	    boost::sort::spreadsort::float_sort(data, data+data_length);
    }
	char flag = 0, flag_;
	float *data2 = (float*) malloc(sizeof(float) * (data_length << 1));
	float *temp = (float*) malloc(sizeof(float) * (data_length << 1));

    float *temp_middle = temp+data_length;
    float *data_middle = data+data_length;

	unsigned int rank_plus_1 = rank+1;
	unsigned int rank_minus_1 = rank-1;
	int determine_ = rank_plus_1<size && rank_plus_1<num;
	int rank_lt_num = rank < num;
	int rank_lt_num_and_rank_neq_0 = rank_lt_num && rank != 0;
	unsigned int data_length_minus_1 = data_length-1;
	if(rank%2){
        while(1){
            flag = 0;
            if(rank_lt_num){
                char change;
                MPI_Isend(data, data_length, MPI_FLOAT, rank_minus_1, 0, MPI_COMM_WORLD, &request);
                MPI_Recv(&change, 1, MPI_CHAR, rank_minus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(change) MPI_Recv(data, data_length, MPI_FLOAT, rank_minus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //char another_flag = 0;
            if(determine_){
                MPI_Recv(data2, recv_data_length, MPI_FLOAT, rank_plus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                float *data_iter=data, *recv_data_iter=data2, *temp_iter=temp;
                float *data_end_iter = data+data_length, *recv_end_iter = data2+recv_data_length;
                if( data[data_length-1] <= *recv_data_iter ) MPI_Isend(&flag, 1, MPI_CHAR, rank_plus_1, 0, MPI_COMM_WORLD, &request);
                else {
                    while( data_iter != data_end_iter && recv_data_iter != recv_end_iter){
                        if(*data_iter <= *recv_data_iter){
                            *temp_iter = *data_iter;
                            temp_iter++;
                            data_iter++;
                        } else{
                            *temp_iter = *recv_data_iter;
                            temp_iter++;
                            recv_data_iter++;
                        }
                    }
                    flag = 1;
                    MPI_Isend(&flag, 1, MPI_CHAR, rank_plus_1, 0, MPI_COMM_WORLD, &request);
                    while( data_iter != data_end_iter ) {
                        *temp_iter = *data_iter;
                        temp_iter++;
                        data_iter++;
                    }
                    while( recv_data_iter != recv_end_iter ){
                        *temp_iter = *recv_data_iter;
                        temp_iter++;
                        recv_data_iter++;
                    }
                    MPI_Isend(temp_middle, recv_data_length, MPI_FLOAT, rank_plus_1, 0, MPI_COMM_WORLD, &request);
                    //for( data_iter = data, temp_iter = temp ; data_iter != data_end_iter ; data_iter++, temp_iter++) *data_iter = *temp_iter;
                    float *save = temp;
                    temp = data;
                    data = save;
                    save = temp_middle;
                    temp_middle = data_middle;
                    data_middle = save;
                }
            }
            MPI_Allreduce(&flag, &flag_, 1, MPI_CHAR, MPI_BOR, MPI_COMM_WORLD);
            if(flag_ == 0) break;
        }
    }
    else{
        while(1){
            flag = 0;
            if(determine_){
                MPI_Recv(data2, recv_data_length, MPI_FLOAT, rank_plus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                float *data_iter=data, *recv_data_iter=data2, *temp_iter=temp;
                float *data_end_iter = data+data_length, *recv_end_iter = data2+recv_data_length;
                if( data[data_length-1] <= *recv_data_iter ) MPI_Isend(&flag, 1, MPI_CHAR, rank_plus_1, 0, MPI_COMM_WORLD, &request);
                else {
                    while( data_iter != data_end_iter && recv_data_iter != recv_end_iter){
                        if(*data_iter <= *recv_data_iter){
                            *temp_iter = *data_iter;
                            ++temp_iter;
                            ++data_iter;
                        }
                        else{
                            *temp_iter = *recv_data_iter;
                            ++temp_iter;
                            ++recv_data_iter;
                        }
                    }
                    flag = 1;
                    MPI_Isend(&flag, 1, MPI_CHAR, rank_plus_1, 0, MPI_COMM_WORLD, &request);
                    while( data_iter != data_end_iter ) {
                        *temp_iter = *data_iter;
                        ++temp_iter;
                        ++data_iter;
                    }
                    while( recv_data_iter != recv_end_iter ){
                        *temp_iter = *recv_data_iter;
                        ++temp_iter;
                        ++recv_data_iter;
                    }
                    MPI_Isend(temp_middle, recv_data_length, MPI_FLOAT, rank_plus_1, 0, MPI_COMM_WORLD, &request);
                    //for( data_iter = data, temp_iter = temp ; data_iter != data_end_iter ; data_iter++, temp_iter++) *data_iter = *temp_iter;
                    float *save = temp;
                    temp = data;
                    data = save;
                    save = temp_middle;
                    temp_middle = data_middle;
                    data_middle = save;
                }
                
            }
            if(rank_lt_num_and_rank_neq_0){
                char change;
                MPI_Isend(data, data_length, MPI_FLOAT, rank_minus_1, 0, MPI_COMM_WORLD, &request);
                MPI_Recv(&change, 1, MPI_CHAR, rank_minus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(change) {
                    MPI_Recv(data, data_length, MPI_FLOAT, rank_minus_1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            
            MPI_Allreduce(&flag, &flag_, 1, MPI_CHAR, MPI_BOR, MPI_COMM_WORLD);
            if(flag_ == 0) break;
            
        }
        
    }
	
    MPI_File f2;
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f2);
	MPI_File_write_at(f2, sizeof(float) * read_pos, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);
	
	MPI_Finalize();
}
