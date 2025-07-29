#include "FileOperations.h"

#include <iostream>

FileOperations::FileOperations(std::string path)
	: path_{path}
{
	isFileExist = false;
	number_of_columns = 0;
}

void FileOperations::create_file(std::string file_name, int number_of_columns) {
	if (!isFileExist) {
		std::string file_path = path_ + "/" + file_name;
		my_file.open(file_path);
		std::cout << "File created: " << file_name << std::endl;
		isFileExist = true;
		this->number_of_columns = number_of_columns;
	}
}

void FileOperations::close_file() {
	if (isFileExist) {
		my_file.close();
		std::cout << "File closed." << std::endl;
	}
}

void FileOperations::append_file_headers(std::string headers) {
	if (isFileExist) {
		my_file << headers << "\n";
		std::cout << "Headers appended to file." << std::endl;
	}
}

void FileOperations::append_file_data(std::vector<double> row, int MATRIX_SIZE) {
	if (isFileExist) {
		for (int i = 0; i < number_of_columns; i++) {
			my_file << row[i] << ",";
		}
		my_file << MATRIX_SIZE << "\n";
		std::cout << "Data appended to file." << std::endl;
	}
}
