#include <string>
#include <fstream>
#include <vector>
#pragma once

class FileOperations {
private:
	bool isFileExist;
	std::ofstream my_file;
	int number_of_columns;

public: 
	FileOperations();

	void create_file(std::string file_name, int number_of_columns);
	void close_file();
	void append_file_data(std::vector<double> row);
	void append_file_headers(std::string headers);
};