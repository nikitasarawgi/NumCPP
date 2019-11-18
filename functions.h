#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <array>
#include "numcpparray.h"
using namespace std;
namespace nc{

	//numpy.zeros
	template<class dtype>
	Ncpparray<dtype> zeros(int rows, int cols = 1){

		Ncpparray<dtype> newArr(rows, cols);
		std::fill(newArr.array, newArr.array + newArr.size, static_cast<dtype>(0));

		return newArr;

	}

	//numpy.ones
	template<class dtype>
	Ncpparray<dtype> ones(int rows, int cols = 1){

		Ncpparray<dtype> newArr(rows, cols);
		std::fill(newArr.array, newArr.array + newArr.size, static_cast<dtype>(1));

		return newArr;

	}

	//numpy.eye
	template<class dtype>
	Ncpparray<dtype> eye(int rows, int cols, int k = 0){

		
		Ncpparray<dtype> newArr = zeros<dtype>(rows,cols);
		if(newArr.shape[1] == 1 || newArr.shape[0] == 1){
			newArr.array[0] = 1;
			return newArr;
		}
		for(int i = 0; i < newArr.shape[1]; i++){
			newArr.array[(i*newArr.shape[1] + i) + k] = 1;
		}
		return newArr;
	}

	//numpy.identity
	template<class dtype>
	Ncpparray<dtype> identity(int rows){

		Ncpparray<dtype> newArr = zeros<dtype>(rows,rows);
		for(int i = 0; i < newArr.shape[0]; i++){
			newArr.array[(i*newArr.shape[1] + i)] = 1;
		}
		return newArr;
	}

	//numpy.empty
	template<class dtype>
	Ncpparray<dtype> empty(int rows, int cols = 1){

		Ncpparray<dtype> newArr(rows,cols);
		return newArr;
	}

	//numpy.maximum
	template<class dtype>
	Ncpparray<dtype> maximum(Ncpparray<dtype> arr1, Ncpparray<dtype> arr2){

		bool great1 = true;
		if(arr1.size != arr2.size){
			cout << "The size needs to be same for comparison!";
			exit(0);
		}
		if(arr1.shape[0] != arr2.shape[0] || arr1.shape[1] != arr2.shape[1]){
			arr2 = arr2.reshape(arr1.shape[0], arr1.shape[1]);
		}

		
		for(int i = 0; i < arr1.shape[0]; i++){
			for(int j = 0; j < arr1.shape[1]; j++){
				if(arr1.array[i * arr1.shape[1] + j] < arr2.array[i * arr1.shape[1] + j]){
					great1 = false;
				}
			}
		}
		if(great1){
			return arr1;
		}
		else{
			return arr2;
		}

	}

	//numpy.minimum
	template<class dtype>
	Ncpparray<dtype> minimum(Ncpparray<dtype> arr1, Ncpparray<dtype> arr2){

		bool great1 = true;
		if(arr1.size != arr2.size){
			cout << "The size needs to be same for comparison!";
			exit(0);
		}
		if(arr1.shape[0] != arr2.shape[0] || arr1.shape[1] != arr2.shape[1]){
			arr2 = arr2.reshape(arr1.shape[0], arr1.shape[1]);
		}

		
		for(int i = 0; i < arr1.shape[0]; i++){
			for(int j = 0; j < arr1.shape[1]; j++){
				if(arr1.array[i * arr1.shape[1] + j] > arr2.array[i * arr1.shape[1] + j]){
					great1 = false;
				}
			}
		}
		if(great1){
			return arr1;
		}
		else{
			return arr2;
		}

	}

	template<class dtype>
	Ncpparray<dtype> amax(Ncpparray<dtype> arr, int axis, bool keepdims){

		Ncpparray<dtype> result;


		if(axis == 0){
			arr = arr.transpose();
			result = empty(1,arr.shape[1]);
		}
		else{
			result = empty(1,arr.shape[0]);
		}
		for(int i = 0; i < arr.shape[0]; i++){
			int resultIndex = std::max_element(arr.array + (i * arr.shape[1]), arr.array + (i * arr.shape[1]) + arr.shape[1]);
			result.array[i] = arr.array[i * arr.shape[1] + resultIndex];
		}
		if(keepdims && axis == 1){
			result = result.reshape(arr.shape[0],1);

		}
		return result;

	}

	template<class dtype>
	Ncpparray<dtype> amin(Ncpparray<dtype> arr, int axis, bool keepdims){

		Ncpparray<dtype> result;


		if(axis == 0){
			arr = arr.transpose();
			result = empty(1,arr.shape[1]);
		}
		else{
			result = empty(1,arr.shape[0]);
		}
		for(int i = 0; i < arr.shape[0]; i++){
			int resultIndex = std::min_element(arr.array + (i * arr.shape[1]), arr.array + (i * arr.shape[1]) + arr.shape[1]);
			result.array[i] = arr.array[i * arr.shape[1] + resultIndex];
		}
		if(keepdims && axis == 1){
			result = result.reshape(arr.shape[0],1);

		}
		return result;

	}

	template<class dtype>
	Ncpparray<dtype> argmax(Ncpparray<dtype> arr, int axis){

		Ncpparray<dtype> result;


		if(axis == 0){
			arr = arr.transpose();
			result = empty(1,arr.shape[1]);
		}
		else{
			result = empty(1,arr.shape[0]);
		}
		for(int i = 0; i < arr.shape[0]; i++){
			int resultIndex = std::max_element(arr.array + (i * arr.shape[1]), arr.array + (i * arr.shape[1]) + arr.shape[1]);
			result.array[i] = resultIndex;
		}
		return result;

	}


	template<class dtype>
	Ncpparray<dtype> argmin(Ncpparray<dtype> arr, int axis){

		Ncpparray<dtype> result;


		if(axis == 0){
			arr = arr.transpose();
			result = empty(1,arr.shape[1]);
		}
		else{
			result = empty(1,arr.shape[0]);
		}
		for(int i = 0; i < arr.shape[0]; i++){
			int resultIndex = std::min_element(arr.array + (i * arr.shape[1]), arr.array + (i * arr.shape[1]) + arr.shape[1]);
			result.array[i] = resultIndex;
		}
		return result;

	}

	template<class dtype>
	dtype sum(Ncpparray<dtype> arr){

		dtype result = std::accumulate(arr.array, arr.array + arr.size, 0);
		return result;
	}

	template<class dtype>
	Ncpparray<dtype> sum(Ncpparray<dtype> arr, int axis, bool keepdims){

		if(axis == 0){
			arr = arr.transpose();
		}

		Ncpparray<dtype> result(1,arr.shape[0]);
		for(int i = 0; i < arr.shape[0]; i++){
			result[i] = std::accumulate(arr.array + (i * arr.size), arr.array + (i * arr.size) + arr.size);
		}

		if(axis == 1 && keepdims){
			result = result.reshape(arr.shape[0], 1);
		}
		return result;
	}


	//adding two matrices
	template<class dtype>
	Ncpparray<dtype> add(Ncpparray<dtype> arr2, Ncpparray<dtype> arr){

		//check which matrix is smaller and then try getting it to the same shape as the other one
		//we have two matrices mat1 and mat2 which will be added later so as to not change exisiting matrices

		Ncpparray<dtype> mat1, mat2;
		bool transposed = false;
		bool thisismat2 = true;


		if(arr2.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr2;
			mat2 = arr;
			thisismat2 = false;
		}
		else if(arr2.shape[1] == 0 && arr2.shape[0] == arr.shape[1]){
			mat1 = arr2.transpose();
			mat2 = arr.transpose();
			transposed = true;
			thisismat2 = false;
		}
		else if(arr.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr;
			mat2 = arr2;
		}
		else if(arr.shape[1] == 1 && arr2.shape[0] == arr.shape[0]){
			mat1 = arr.transpose();
			mat2 = arr2.transpose();
			transposed = true;
		}
		else if(arr.shape[0] == arr2.shape[0] && arr.shape[1] == arr2.shape[1]){
			mat1 = arr2;
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
	template<class dtype>
	Ncpparray<dtype> subtract(Ncpparray<dtype> arr2, Ncpparray<dtype> arr){

		//check which matrix is smaller and then try getting it to the same shape as the other one
		//we have two matrices mat1 and mat2 which will be added later so as to not change exisiting matrices

		Ncpparray<dtype> mat1, mat2;
		bool transposed = false;
		bool thisismat2 = true;


		if(arr2.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr2;
			mat2 = arr;
			thisismat2 = false;
		}
		else if(arr2.shape[1] == 0 && arr2.shape[0] == arr.shape[1]){
			mat1 = arr2.transpose();
			mat2 = arr.transpose();
			transposed = true;
			thisismat2 = false;
		}
		else if(arr.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr;
			mat2 = arr2;
		}
		else if(arr.shape[1] == 1 && arr2.shape[0] == arr.shape[0]){
			mat1 = arr.transpose();
			mat2 = arr2.transpose();
			transposed = true;
		}
		else if(arr.shape[0] == arr2.shape[0] && arr.shape[1] == arr2.shape[1]){
			mat1 = arr2;
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

	//numpy.multiply
	//Element wise multiplaction
	template<class dtype>
	Ncpparray<dtype> multiply(Ncpparray<dtype> arr2, Ncpparray<dtype> arr){

		//check which matrix is smaller and then try getting it to the same shape as the other one
		//we have two matrices mat1 and mat2 which will be added later so as to not change exisiting matrices

		Ncpparray<dtype> mat1, mat2;
		bool transposed = false;
		bool thisismat2 = true;


		if(arr2.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr2;
			mat2 = arr;
			thisismat2 = false;
		}
		else if(arr2.shape[1] == 0 && arr2.shape[0] == arr.shape[1]){
			mat1 = arr2.transpose();
			mat2 = arr.transpose();
			transposed = true;
			thisismat2 = false;
		}
		else if(arr.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr;
			mat2 = arr2;
		}
		else if(arr.shape[1] == 1 && arr2.shape[0] == arr.shape[0]){
			mat1 = arr.transpose();
			mat2 = arr2.transpose();
			transposed = true;
		}
		else if(arr.shape[0] == arr2.shape[0] && arr.shape[1] == arr2.shape[1]){
			mat1 = arr2;
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
					
					mat2.array[i * mat2.shape[1] + j] *= mat1.array[i * mat1.shape[1] + j];
					

			}
		}

		return mat2;
	}

	template<class dtype>
	Ncpparray<dtype> multiply(Ncpparray<dtype> arr, dtype x){

		Ncpparray<dtype> result = arr;
		for(int i = 0; i < result.shape[0]; i++){
			for(int j = 0; j < result.shape[1]; j++){
				result.array[i * result.shape[1] + j] *= x;
			}
		}
		return result;
	}

	//divide two matrices
	template<class dtype>
	Ncpparray<dtype> divide(Ncpparray<dtype> arr2, Ncpparray<dtype> arr){

		//check which matrix is smaller and then try getting it to the same shape as the other one
		//we have two matrices mat1 and mat2 which will be added later so as to not change exisiting matrices

		Ncpparray<dtype> mat1, mat2;
		bool transposed = false;
		bool thisismat2 = true;


		if(arr2.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr2;
			mat2 = arr;
			thisismat2 = false;
		}
		else if(arr2.shape[1] == 0 && arr2.shape[0] == arr.shape[1]){
			mat1 = arr2.transpose();
			mat2 = arr.transpose();
			transposed = true;
			thisismat2 = false;
		}
		else if(arr.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr;
			mat2 = arr2;
		}
		else if(arr.shape[1] == 1 && arr2.shape[0] == arr.shape[0]){
			mat1 = arr.transpose();
			mat2 = arr2.transpose();
			transposed = true;
		}
		else if(arr.shape[0] == arr2.shape[0] && arr.shape[1] == arr2.shape[1]){
			mat1 = arr2;
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
					
					mat2.array[i * mat2.shape[1] + j] /= mat1.array[i * mat1.shape[1] + j];
					

			}
		}

		return mat2;
	}

	template<class dtype>
	Ncpparray<dtype> divide(Ncpparray<dtype> arr, dtype x){

		Ncpparray<dtype> result = arr;
		for(int i = 0; i < result.shape[0]; i++){
			for(int j = 0; j < result.shape[1]; j++){
				result.array[i * result.shape[1] + j] /= x;
			}
		}
		return result;
	}

	//numpy.equal
	template<class dtype>
	Ncpparray<bool> equal(Ncpparray<dtype> arr2, Ncpparray<dtype> arr){

		//check which matrix is smaller and then try getting it to the same shape as the other one
		//we have two matrices mat1 and mat2 which will be added later so as to not change exisiting matrices

		Ncpparray<dtype> mat1, mat2;
		bool transposed = false;
		bool thisismat2 = true;


		if(arr2.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr2;
			mat2 = arr;
			thisismat2 = false;
		}
		else if(arr2.shape[1] == 0 && arr2.shape[0] == arr.shape[1]){
			mat1 = arr2.transpose();
			mat2 = arr.transpose();
			transposed = true;
			thisismat2 = false;
		}
		else if(arr.shape[0] == 1 && arr2.shape[1] == arr.shape[1]){
			mat1 = arr;
			mat2 = arr2;
		}
		else if(arr.shape[1] == 1 && arr2.shape[0] == arr.shape[0]){
			mat1 = arr.transpose();
			mat2 = arr2.transpose();
			transposed = true;
		}
		else if(arr.shape[0] == arr2.shape[0] && arr.shape[1] == arr2.shape[1]){
			mat1 = arr2;
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
		Ncpparray<bool> result(mat1.shape[0], mat1.shape[1]);
		for(int i = 0; i < mat1.shape[0]; i++){
			for(int j = 0; j < mat1.shape[1]; j++){
					
					if(mat2.array[i * mat2.shape[1] + j] == mat1.array[i * mat1.shape[1] + j]){
						result.array[i * result.shape[1] + j] = true;
					}
					else{
						result.array[i * result.shape[1] + j] = false;
					}
					

			}
		}

		return result;
	}


}
