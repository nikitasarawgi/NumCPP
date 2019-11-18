// file that defines the numpy array class and associated properties

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <array>
using namespace std;
namespace nc{
	template <class dtype>
	class Ncpparray{

	public:

		int shape[2]{0,0};
		int size{0};
		dtype* array{nullptr};

	public:

		//Basic Constructor class
		Ncpparray(){

		}

		//Constructor for a square sized matrix
		Ncpparray(int squareSize){
			shape[0] = squareSize;
			shape[1] = squareSize;
			size = squareSize * squareSize;
			array = new dtype[size];
		}

		//Constructor for matrix with rows and columns
		Ncpparray(int rows, int cols):size(rows * cols),array(new dtype[rows*cols]){
			shape[0] = rows;
			shape[1] = cols;
			
		}

		//Copy Constructor
		Ncpparray(const Ncpparray<dtype> &arr){
			//cout << "This is called";
			shape[0] = arr.shape[0];
			shape[1] = arr.shape[1];
			size = arr.size;
			array = new dtype[size];
			std::copy(arr.array,arr.array + size, array);
			//array = arr2.array;
		}

		~Ncpparray(){
			delete[] array;
		}

		Ncpparray<dtype> operator=(Ncpparray<dtype> arr){
			size = arr.size;
			shape[0] = arr.shape[0];
			shape[1] = arr.shape[1];
			array = new dtype[size];
			std::copy(arr.array,arr.array + size, array);
			return *this;
		}

		

		void print(){

			for(int i = 0; i < shape[0]; i++){
				for(int j = 0; j < shape[1]; j++){
					cout << array[i * shape[1] + j] << " ";
				}
				cout << "\n";
			}
			cout << "\n";
		}


		//reshape the array
		Ncpparray<dtype> reshape(int rows, int cols){

			if(rows*cols != size){
				cout << "Cannot be reshaped!";
			}

			Ncpparray<dtype> newArr = *this;
			
			newArr.shape[0] = rows;
			newArr.shape[1] = cols;
			
			return newArr;
		}

		//Transpose of a matrix
		Ncpparray<dtype> transpose(){

			Ncpparray<dtype> newArr(shape[1],shape[0]);
			for(int i = 0; i < shape[0]; i++){
				for(int j = 0; j < shape[1]; j++){
					newArr.array[j * newArr.shape[1] + i] = array[i * shape[1] + j];
				}
			}
			return newArr;

		}

		//adding two matrices
		Ncpparray<dtype> operator +(Ncpparray<dtype> arr){

			//check which matrix is smaller and then try getting it to the same shape as the other one
			//we have two matrices mat1 and mat2 which will be added later so as to not change exisiting matrices

			Ncpparray<dtype> mat1, mat2;
			bool transposed = false;
			bool thisismat2 = true;


			if(shape[0] == 1 && shape[1] == arr.shape[1]){
				mat1 = *this;
				mat2 = arr;
				thisismat2 = false;
			}
			else if(shape[1] == 0 && shape[0] == arr.shape[1]){
				mat1 = transpose();
				mat2 = arr.transpose();
				transposed = true;
				thisismat2 = false;
			}
			else if(arr.shape[0] == 1 && shape[1] == arr.shape[1]){
				mat1 = arr;
				mat2 = *this;
			}
			else if(arr.shape[1] == 1 && shape[0] == arr.shape[0]){
				mat1 = arr.transpose();
				mat2 = transpose();
				transposed = true;
			}
			else if(arr.shape[0] == shape[0] && arr.shape[1] == shape[1]){
				mat1 = *this;
				mat2 = arr;
			}
			else{
				cout << "operands could not be broadcasted together!\n";
				exit(0);
			}


			if(mat1.size < mat2.size){

				
				Ncpparray<dtype> newmat1(mat2.shape[0], mat2.shape[1]);
				//Copy the mat1 matrix;
				
				for(int i = 0; i < mat2.shape[0] / mat1.shape[0]; i++){
					std::copy(mat1.array,mat1.array + mat1.size, newmat1.array + (i * mat1.size));
				}
				
				mat1 = newmat1;

				if(transposed){
					mat1 = mat1.transpose();
					mat2 = mat2.transpose();
				}

				if(!thisismat2){
					newmat1 = mat1;
					mat1 = mat2;
					mat2 = newmat1;
				}

			}

			//We now simply have to add current array and new modified array1 which are supposed to be of the same size
			for(int i = 0; i < mat1.shape[0]; i++){
				for(int j = 0; j < mat1.shape[1]; j++){
					
						mat2.array[i * mat2.shape[1] + j] += mat1.array[i * mat1.shape[1] + j];
					

				}
			}



			return mat2;
		}


		//subtracting two matrices
		Ncpparray<dtype> operator -(Ncpparray<dtype> &arr){

			//check which matrix is smaller and then try getting it to the same shape as the other one
			//we have two matrices mat1 and mat2 which will be added later so as to not change exisiting matrices

			Ncpparray<dtype> mat1, mat2;
			bool transposed = false;
			bool thisismat2 = true;


			if(shape[0] == 1 && shape[1] == arr.shape[1]){
				mat1 = *this;
				mat2 = arr;
				thisismat2 = false;
			}
			else if(shape[1] == 0 && shape[0] == arr.shape[1]){
				mat1 = transpose();
				mat2 = arr.transpose();
				transposed = true;
				thisismat2 = false;
			}
			else if(arr.shape[0] == 1 && shape[1] == arr.shape[1]){
				mat1 = arr;
				mat2 = *this;
			}
			else if(arr.shape[1] == 1 && shape[0] == arr.shape[0]){
				mat1 = arr.transpose();
				mat2 = transpose();
				transposed = true;
			}
			else if(arr.shape[0] == shape[0] && arr.shape[1] == shape[1]){
				mat1 = *this;
				mat2 = arr;
			}
			else{
				cout << "operands could not be broadcasted together!\n";
				exit(0);
			}

			if(mat1.size < mat2.size){

				
				Ncpparray<dtype> newmat1(mat2.shape[0], mat2.shape[1]);
				//Copy the mat1 matrix;
				
				for(int i = 0; i < mat2.shape[0] / mat1.shape[0]; i++){
					std::copy(mat1.array,mat1.array + mat1.size, newmat1.array + (i * mat1.size));
				}
				
				mat1 = newmat1;

				if(transposed){
					mat1 = mat1.transpose();
					mat2 = mat2.transpose();
				}

				if(!thisismat2){
					newmat1 = mat1;
					mat1 = mat2;
					mat2 = newmat1;
				}

			}

			//We now simply have to add current array and new modified array1 which are supposed to be of the same size
			for(int i = 0; i < mat1.shape[0]; i++){
				for(int j = 0; j < mat1.shape[1]; j++){
					
						mat2.array[i * mat2.shape[1] + j] -= mat1.array[i * mat1.shape[1] + j];
					

				}
			}

			return mat2;
		}

		Ncpparray<dtype> dot(Ncpparray<dtype> arr){

			Ncpparray<dtype> mat1,mat2;
			if(shape[1] != arr.shape[0]){
				cout <<"Cannot be multiplied because dimensions do not match!\n";
				exit(0);
			}
			mat1 = *this;
			mat2 = arr;
			Ncpparray<dtype> result(mat1.shape[0], mat2.shape[1]);
			std::fill(result.array, result.array + result.size, static_cast<dtype>(0));

			for(int i = 0; i < mat1.shape[0]; i++){
				for(int j = 0; j < mat2.shape[1]; j++){
					
					for(int k = 0; k < mat1.shape[1]; k++){
						result.array[i * result.shape[1] + j] += mat1.array[i * mat1.shape[1] + k] * mat2.array[k * mat2.shape[1] + j]; 
					}
				}
			}
			return result;
		}

		dtype operator[](int index){

			cout << "SIZE: " << size << "\n";

			if(index >= size){
				cout << "Array index out of bounds!\n";
				exit(0);
			}

			if(index < 0){
				return array[size + index];
			}

			return array[index];
		}

		dtype operator()(int i, int j){

			if(i == 1 & shape[0] == 1){
				return array[j];
			}
			if(j == 1 && shape[1] == 1){
				return array[i];
			}

			if(i >= shape[0] || j >= shape[1]){
				cout << "Array index out of bounds!\n";
				exit(0);
			}
			if(i < 0){
				i += shape[0];
			}
			if(j < 0){
				j += shape[1];
			}

			return array[i * shape[1] + j];
		}

		Ncpparray<dtype> dot(dtype x){

			Ncpparray<dtype> result = *this;
			for(int i = 0; i < result.shape[0]; i++){
				for(int j = 0; j < result.shape[1]; j++){
					result.array[i * result.shape[1] + j] *= x;
				}
			}
			return result;
		}

		

	};
}