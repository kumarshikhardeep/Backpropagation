/*This is a back-propagation algorithm using gradient descent and we have
trained the feed-forward neural network for N bit palindrome.
There is 1 input layer which can contain any number of input neurons as per the user's choice,
1 hidden layer containing which can contain any number of neurons as well as per the user's choice
and 1 output layer containing 1 neuron.
We are iterating "times" number of times until the error we are getting is less than the desired error limit
and the output we are getting are almost equal to the desired output.
We are treating a number as 0 is it is less than 0.5 and 1 otherwise, based on this assumption we have trained
the neural network to detect the palindrome pattern and gives the output close to 1 if true and close to 0 if false.
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define InputN 4		// number of neurons in the input layer
#define HN 2			// number of neurons in the hidden layer
#define OutN 1			// number of neurons in the output layer
#define datanum 3		// number of training samples

void main(){
	double sigmoid(double);
	char* result;
	double x_out[InputN];		// input layer
	double hn_out[HN];			// hidden layer
	double y_out[OutN];         // output layer
	double y[OutN];				// expected output layer
	double w[InputN][HN];		// weights from input layer to hidden layer
	double v[HN][OutN];			// weights from hidden layer to output layer

	double deltaw[InputN][HN];
	double deltav[HN][OutN];

	double hn_delta[HN];		// delta of hidden layer
	double y_delta[OutN];		// delta of output layer
	double error;
	double errlimit = 0.00001;
	double alpha = 0.1;
	double beta = 0.5;
	int loop = 0;
	int times = 100000;
	int i, j, m;
	double max, min;
	double sumtemp;
	double errtemp;
	int flag,k;

	// training set
	struct{
		double input[InputN];
		double teach[OutN];
	}data[datanum];

	// Generate data samples
	for(m=0; m<datanum; m++){
		for(i=0; i<InputN; i++)
			data[m].input[i] = (double)rand()/32767.0;
        flag=1;
        for(i=0;i<InputN/2;i++)
        {
            if((data[m].input[i]>=0.5 && data[m].input[InputN-i-1]>=0.5 ) || (data[m].input[i]<0.5  && data[m].input[InputN-i-1]<0.5))
             {
                 continue;
             }
             else
             {
                flag=0;
             }
        }
        if(flag==1)
        {
             double x= (double)rand()/32767.0;
                  if(x<0.5)
                      x=x+0.5;
                  data[m].teach[0]=x;
        }
        else
        {
            double x= (double)rand()/32767.0;
                  if(x>0.5)
                      x=x-0.5;
                  data[m].teach[0]=x;
        }
	}


	// Initializition
	for(i=0; i<InputN; i++){
		for(j=0; j<HN; j++){
		    w[i][j]=0;
			deltaw[i][j] = 0;
		}
	}
	for(i=0; i<HN; i++){
		for(j=0; j<OutN; j++){
			v[i][j]=0;
			deltav[i][j] = 0;
		}
	}

	// Training
	while(loop < times){
		loop++;
		error = 0.0;

		for(m=0; m<datanum ; m++){

			// Feedforward

			max = 0.0;
			min = 0.0;
			for(i=0; i<InputN; i++){
				x_out[i] = data[m].input[i];
				if(max < x_out[i])
					max = x_out[i];
				if(min > x_out[i])
					min = x_out[i];
			}
			for(i=0; i<InputN; i++){
				x_out[i] = (x_out[i] - min) / (max - min);
			}

			for(i=0; i<OutN ; i++){
				y[i] = data[m].teach[i];
			}

			for(i=0; i<HN; i++){
				sumtemp = 0.0;
				for(j=0; j<InputN; j++)
					sumtemp += w[j][i] * x_out[j];
				hn_out[i] = sigmoid(sumtemp);		// sigmoid serves as the activation function
			}

			for(i=0; i<OutN; i++){
				sumtemp = 0.0;
				for(j=0; j<HN; j++)
					sumtemp += v[j][i] * hn_out[j];
				y_out[i] = sigmoid(sumtemp);

			}
			for(k=0;k<InputN;k++)
            {
                printf("%f  ", data[m].input[k]);
            }
            printf("%f(expected)  ",data[m].teach[0]);
			printf("%f(actual) \n",y_out[0]);

			// Backpropagation
			for(i=0; i<OutN; i++){
				errtemp = y[i] - y_out[i];
				y_delta[i] = -errtemp * y_out[i] * (1.0 - y_out[i]);
				error += errtemp*errtemp;
			}

			for(i=0; i<HN; i++){
				errtemp = 0.0;
				for(j=0; j<OutN; j++)
					errtemp += y_delta[j] * v[i][j];
				hn_delta[i] = errtemp * ( hn_out[i]) * (1.0 - hn_out[i])*data[m].input[i];
			}

			// Stochastic gradient descent
			for(i=0; i<OutN; i++){
				for(j=0; j<HN; j++){
					deltav[j][i] = alpha*deltav[j][i]+ beta * y_delta[i] * hn_out[j];
					v[j][i] -= deltav[j][i];
				}
			}

			for(i=0; i<HN; i++){
				for(j=0; j<InputN; j++){
					deltaw[j][i] = alpha*deltaw[j][i]+  beta * hn_delta[i] * x_out[j];
					w[j][i] -= deltaw[j][i];
				}
			}
		}

		error = error / 2;
		if(error < errlimit)
			break;

		printf("The %d th training, error: %f\n", loop, error);
	}
printf("The %d th training, error: %f\n", loop, error);
}

// sigmoid serves as avtivation function
double sigmoid(double x){
	return(1.0 / (1.0 + exp(-x)));
}
