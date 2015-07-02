// caffe-main.cpp : Defines the entry point for the application.
//

//#include "stdafx.h"


int caffe_main(int argc, char** argv);
int convert_mnist_data_main(int argc, char** argv);
int convert_cifar10_data_main(int argc, char** argv);
int convert_cifar100_data_main(int argc, char** argv);
int compute_image_mean_main(int argc, char** argv);
int convert_imageset_main(int argc, char** argv);
int device_query_main(int argc, char** argv);
int dump_network_main(int argc, char** argv);
int extract_features_main(int argc, char** argv);
int finetune_net_main(int argc, char** argv);
int net_speed_benchmark_main(int argc, char** argv);
int test_net_main(int argc, char** argv);
int train_net_main(int argc, char** argv);
int upgrade_net_proto_binary_main(int argc, char** argv);
int upgrade_net_proto_text_main(int argc, char** argv);



int main(int argc, char** argv) {
	caffe_main(argc, argv);
	//convert_mnist_data_main(argc, argv);
	//convert_cifar10_data_main(argc, argv);
	//convert_cifar100_data_main(argc, argv);
	//compute_image_mean_main(argc, argv);
	//convert_imageset_main(argc, argv);
	//device_query_main(argc, argv);
	//dump_network_main(argc, argv);
	//extract_features_main(argc, argv);
	//finetune_net_main(argc, argv);
	//net_speed_benchmark_main(argc, argv);
	//test_net_main(argc, argv);
	//train_net_main(argc, argv);
	//upgrade_net_proto_binary_main(argc, argv);
	//upgrade_net_proto_text_main(argc, argv);
}
