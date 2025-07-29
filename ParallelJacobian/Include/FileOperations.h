#pragma once

#include <fstream>
#include <string>
#include <vector>

class FileOperations {
private:
	bool isFileExist;
	std::ofstream my_file;
	int number_of_columns;
	std::string path_;

public: 
	FileOperations(std::string path);

	void create_file(std::string file_name, int number_of_columns);
	void close_file();
	void append_file_data(std::vector<double> row, int MATRIX_SIZE);
	void append_file_headers(std::string headers);
};
