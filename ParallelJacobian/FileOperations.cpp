#include "FileOperations.h"
#include <iostream>

FileOperations::FileOperations() {
	isFileExist = false;
	number_of_columns = 0;
};

void FileOperations::create_file(std::string file_name, int number_of_columns) {
	if (!isFileExist) {
		my_file.open(file_name);
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

void FileOperations::append_file_data(std::vector<double> row) {
	if (isFileExist) {
		for (int i = 0; i < number_of_columns; i++) {
			my_file << row[i];
			if (i != number_of_columns - 1) {
				my_file << ",";
			}
			else {
				my_file << "\n";
			}
		}
		std::cout << "Data appended to file." << std::endl;
	}
}