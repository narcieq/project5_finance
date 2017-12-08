/*
#include "mpi.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <vector>
//#include <armadillo>
#include <string>
using namespace  std;
//using namespace arma;
// output file
ofstream ofile;

double ran2(long *);

// Function to initialise energy and magnetization
void InitializeLattice(int, double*);
// The metropolis algorithm including the loop over Monte Carlo cycles
void MetropolisSampling(int, int, double*, double, double*, long&, int, int);
// prints to file the results of the calculations
void WriteResultstoFile(int, int, double*, double*, double);
void Transaction(double* agentMatrix, int trans_agent1, int trans_agent2);

// Main program begins here

int main(int argc, char* argv[])
{
	string filename;
	int agent, MCcycles;
	if (argc <= 2) {
		cout << "Bad Usage: " << argv[0] <<
			" read output file, Number of agents, MC cycles" << endl;
		exit(1);
	}
	if (argc > 1) {
		filename = argv[1];
		agent = atoi(argv[2]);
		MCcycles = atoi(argv[3]);
	}
	// Declare new file name and add lattice size to file name
	string fileout = filename;
	string argument = to_string(agent);
	fileout.append(argument);
	argument = to_string(MCcycles);
	fileout += "_";
	fileout.append(argument);
	fileout += ".txt";
	ofile.open(fileout);

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);



	cout << "processor name : " << processor_name << endl;
	cout << "rank : " << world_rank << endl;
	cout << "size : " << world_size << endl;

	// Start Monte Carlo sampling by looping over the selcted Temperatures
	double* TransactionResult = new double[6];
	double TotalExpectation[6];
	for (int i = 0; i < 6; i++) {
		TransactionResult[i] = 0;
		TotalExpectation[i] = 0;
	}

	int no_intervalls = MCcycles / world_size;
	int myloop_begin = world_rank*no_intervalls + 1;
	int myloop_end = (world_rank + 1)*no_intervalls;
	if ((world_rank == world_size - 1) && (myloop_end < MCcycles)) myloop_end = MCcycles;

	// Start Monte Carlo computation and get expectation values
	MPI_Bcast(&agent, 1, MPI_INT, 0, MPI_COMM_WORLD);

	long idum = -1 - world_rank;

	double  TimeStart, TimeEnd, TotalTime;
	TimeStart = MPI_Wtime();

	double delta_m = 0.05;
	double* count = new double[agent / delta_m];
	for (int i = 0; i < agent / delta_m; i++) {
		count[i] = 0;
	}

	MetropolisSampling(agent, MCcycles, count, delta_m, TransactionResult, idum, myloop_begin, myloop_end);

	cout << "MCS finished" << endl;

	for (int i = 0; i < 6; i++) {
		int a = MPI_Reduce(&TransactionResult[i], &TotalExpectation[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	}
	// print results
	if (world_rank == 0) {
		WriteResultstoFile(agent, MCcycles, TotalExpectation, count, delta_m);
	}



	cout << "rank : " << world_rank << endl;
	cout << "size : " << world_size << endl;

	TimeEnd = MPI_Wtime();
	TotalTime = TimeEnd = TimeStart;
	cout << "Time = " << TotalTime << endl;

	ofile.close();  // close output file

	MPI_Finalize();


	return 0;
}



// The Monte Carlo part with the Metropolis algo with sweeps over the lattice
void MetropolisSampling(int agent, int MCcycles, double* count, double delta_m, double* TransactionResult, long& idum, int myloop_begin, int myloop_end)
{

	// Initialize the agent values
	double* agentMatrix = new double[agent];


	// initialize array for expectation values
	InitializeLattice(agent, agentMatrix);


	// Start Monte Carlo cycles
	int trans_agent1 = 0;
	int trans_agent2 = 0;
	double temp, temp2;
	double Result;

	for (int cycles = myloop_begin; cycles <= myloop_end; cycles++) {
		cout << cycles << "cycles " << endl;
		Result = 0;
		// Start transaction of agents
		for (int x = 0; x < 1.0e7; x++) {
			//pick two different agent
			while (trans_agent1 == trans_agent2) {
				random_device seed;
				mt19937 engine(seed());
				uniform_int_distribution<int> values(0, agent-1);
				trans_agent1 = (int)values(engine);

				random_device seed2;
				mt19937 engine2(seed2());
				uniform_int_distribution<int> values2(0, agent-1);
				trans_agent2 = (int)values2(engine2);
			}

			Transaction(agentMatrix, trans_agent1, trans_agent2);

			Result += agentMatrix[trans_agent1];
			Result += agentMatrix[trans_agent2];

			trans_agent1 = 0;
			trans_agent2 = 0;
		}


		// update result values for local node
		TransactionResult[0] += Result / 1.0e7;



	}
	for (int i = 0; i < agent; i++) {
		int  c = (int)(agentMatrix[agent] / delta_m);
		count[c] += 1;
	}



} // end of Metropolis sampling over spins


void Transaction(double* agentMatrix, int trans_agent1, int trans_agent2) {
	random_device seed3;
	mt19937 engine3(seed3());
	uniform_real_distribution<double> values3(0.0, 1.0);
	double epsilon = values3(engine3);
	double m_i = agentMatrix[trans_agent1];
	double m_j = agentMatrix[trans_agent2];
	double total_money = m_i + m_j;
	agentMatrix[trans_agent1] = epsilon * total_money;
	agentMatrix[trans_agent2] = (1.0 - epsilon) * total_money;
	if (agentMatrix[trans_agent1] < 0.0 || agentMatrix[trans_agent2] < 0.0) {
		agentMatrix[trans_agent1] = m_i;
		agentMatrix[trans_agent1] = m_j;
	}
}

void InitializeLattice(int agent, double* agentMatrix)
{
	for (int i = 0; i < agent; i++) {
		agentMatrix[i] = 1.0;
	}
}// end function initialise



void WriteResultstoFile(int agent, int MCcycles, double* TransactionResult, double* count, double delta_m)
{

	double expectation_m_TransactionResult = TransactionResult[0];

	// all expectation values are per spin, divide by 1/agent/agent
	ofile << setiosflags(ios::showpoint | ios::uppercase);
	// <m>
	ofile << setw(15) << setprecision(8) << expectation_m_TransactionResult / agent << endl;

	ofile << "m + delta_m " << setw(15) << "count" << endl;
	for (int i = 0; i < MCcycles; i++) {
		ofile << setprecision(8) << (double)i / delta_m << " " << count[i] << endl;
	}




}

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double ran2(long *idum)
{
	int            j;
	long           k;
	static long    idum2 = 123456789;
	static long    iy = 0;
	static long    iv[NTAB];
	double         temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum = 1;
		else             *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB + 7; j >= 0; j--) {
			k = (*idum) / IQ1;
			*idum = IA1*(*idum - k*IQ1) - k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB)  iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1;
	*idum = IA1*(*idum - k*IQ1) - k*IR1;
	if (*idum < 0) *idum += IM1;
	k = idum2 / IQ2;
	idum2 = IA2*(idum2 - k*IQ2) - k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j = iy / NDIV;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp = AM*iy) > RNMX) return RNMX;
	else return temp;
}
#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
*/

/*
Program to simulate financial transactions among financial agents using Monte Carlo methods.
MCs 1e3~1e4
transactions:1e7
agents: N=500
Initial money for each agents is m=1
dm is set in the range of [0.01,0.05]
using MPI to speed up
Metropolis sampling is used. Periodic boundary conditions.
The code needs an output file on the command line and the variables
When running MPI,the command arguments is like this:
-n 4 Pro4Yisha.exe L40_MPI_MCs 40 1000000 2.0 2.4 0.05
(-n,number of processes you used,the name of the project file,
outputfile name,lattice,MCs,initial temp, final temp and temp step).
*/

#include "mpi.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
//#include <armadillo>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
using namespace  std;
//using namespace arma;

struct agent_pair {
	int i;
	int j;
};

// output file
ofstream ofile;

// inline function for periodic boundary conditions
inline int periodic(int i, int limit, int add) {
	return (i + limit + add) % (limit);
}
// Function to initialise agent_matrix
void initialize(int, double*);
// The metropolis algorithm
void Metropolis(int, int, long&, double *, double&, double, int, double*);
void Metropolis_d(int, int, long&, double *, double&, double, int, double*,double);
// prints to file the results of the calculations
void output(int, int, double, int, double, double*);
//  Matrix memory allocation
//  allocate space for a matrix
void  **matrix(int, int, int);
//  free space for  a matrix
void free_matrix(void **);
// ran2 for uniform deviates, initialize with negative seed.
double ran2(long *);
agent_pair ran_pair(int n_agents, long&);
agent_pair ran_pair_nearest(int n_agents, long& idum, double alpha, double*);

//**************** Main program begins here

int main(int argc, char* argv[])
{
	string outfilename;
	long idum;
	int n_agents, n_trans, mcs, my_rank, numprocs;
	double *agent_matrix, *counter, *total_counter, average, total_average, M, dm;

	//  MPI initializations
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (my_rank == 0 && argc <= 1) {
		cout << "Bad Usage: " << argv[0] <<
			" read output file" << endl;
		exit(1);
	}

	//initialize from command line
	if (my_rank == 0 && argc > 1) {
		outfilename = argv[1];
	}
	n_agents = atoi(argv[2]); //initialize as 500
	n_trans = atoi(argv[3]); //initialize as 1e7
	mcs = atoi(argv[4]); //initalize as 1e3~1e4
	dm = (double)atof(argv[5]);//0.01~0.05

							   // Declare new file name and add MCs to file name
	string fileout = outfilename;
	string argument = to_string(mcs);
	fileout.append(argument);
	fileout.append(".txt");
	ofile.open(fileout);
	/*
	Determine number of intervall which are used by all processes
	myloop_begin gives the starting point on process my_rank
	myloop_end gives the end point for summation on process my_rank
	*/
	int no_intervalls = mcs / numprocs;
	int myloop_begin = my_rank*no_intervalls + 1;
	int myloop_end = (my_rank + 1)*no_intervalls;
	if ((my_rank == numprocs - 1) && (myloop_end < mcs)) myloop_end = mcs;

	// broadcast to all nodes common variables????
	MPI_Bcast(&n_agents, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_trans, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//  Allocate memory for agent matrix
	agent_matrix = new double[n_agents];
	//Initialize agents_matrix and average values
	initialize(n_agents, agent_matrix);
	M = 0.;//money in the whole system
	average = 0.;
	total_average = 0.;

	int factorMax = ceil(n_agents*1.0 / dm); //number of dm
											 // Allocate memory for counter and initialize as 0
	counter = new double[factorMax + 1]();
	total_counter = new double[factorMax + 1]();
	// every node has its own seed for the random numbers, this is important else
	// if one starts with the same seed, one ends with the same random numbers

	idum = -1 - my_rank;  // random starting point
	double  TimeStart, TimeEnd, TotalTime;
	TimeStart = MPI_Wtime();

	string flag;
	flag = argv[6];

	// start Monte Carlo computation
		// check flag
	for (int i = 1; i < argc; i++) {
		// task a
		if ((string(argv[i]).find("-") == 0 && string(argv[i]).find("a") != string::npos)) {
			for (int cycles = myloop_begin; cycles <= myloop_end; cycles++) {
				Metropolis(n_agents, n_trans, idum, agent_matrix, M, dm, factorMax, counter);
				// update expectation values  for local node
				average += M;
				M = 0;
			}
		}

		//task d
		else if ((string(argv[i]).find("-") == 0 && string(argv[i]).find("d") != string::npos)) {
			double alpha = atof(argv[7]);
			for (int cycles = myloop_begin; cycles <= myloop_end; cycles++) {
				Metropolis_d(n_agents, n_trans, idum, agent_matrix, M, dm, factorMax, counter, alpha);
				// update expectation values  for local node
				average += M;
				M = 0;
			}
		}
	}

	//count times for each dm


	// Find total average and count times
	MPI_Reduce(&average, &total_average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//      for(int i=0;i<=factorMax;i++)
	//      MPI_Reduce(&counter[i], &total_counter[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	const int size = factorMax + 1;
	MPI_Reduce(counter, total_counter, size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	cout << "total_counter[1]" << total_counter[1];
	cout << "counter[1]" << counter[1] << endl;
	// print results
	if (my_rank == 0) {
		output(n_agents, mcs, total_average, factorMax, dm, total_counter);

	}


	delete[] agent_matrix; // free memory
	delete[] counter;
	delete[] total_counter;
	ofile.close();  // close output file
	TimeEnd = MPI_Wtime();
	TotalTime = TimeEnd - TimeStart;
	if (my_rank == 0) {
		cout << "Time = " << TotalTime << " on number of processors: " << numprocs << endl;
	}

	// End MPI
	MPI_Finalize();
	return 0;
} // end main function


  // function to initialise agent_matrix
void initialize(int n_agents, double *agent_matrix)
{
	// setup agent matrix and intial money
	for (int x = 0; x < n_agents; x++)
		agent_matrix[x] = 1.0; // initial money for each agent 1.0

}// end function initialise

 //***              Metropolis algo
 //calculate total money in the whole system and
 //count times for dm,2dm,3dm appeared in each cycle
void Metropolis(int n_agents, int n_trans, long& idum, double *agent_matrix, double& M, double dm, int factorMax, double* counter)
{
	double epsilon, totalM2;
	int factor, i, j;
	// loop over random agent pair with 1e7 transactions
	for (int t = 0; t < n_trans; t++) {
		//randomly pick a pair of agents
		agent_pair agentPair = ran_pair(n_agents, idum);
		i = agentPair.i;
		j = agentPair.j;
		//calculate original total money of the two agents
		totalM2 = agent_matrix[i] + agent_matrix[j];
		//calculate new income afer transaction
		epsilon = ran2(&idum);
		agent_matrix[i] = epsilon*totalM2;
		agent_matrix[j] = (1 - epsilon)*totalM2;
	}

	for (int i = 0; i < n_agents; i++) {
		M += agent_matrix[i];
		factor = ceil(agent_matrix[i] / dm);
		for (int j = 1; j <= factorMax; j++) {
			if (factor == j) {
				counter[j] += 1.0;
			}
		} //counting distibution of money after 1e7 transactions
	}


} // end of Metropolis sampling over agent pair

void Metropolis_d(int n_agents, int n_trans, long& idum, double *agent_matrix, double& M, double dm, int factorMax, double* counter, double alpha)
{
	double epsilon, totalM2;
	int factor, i, j;
	// loop over random agent pair with 1e7 transactions
	for (int t = 0; t < n_trans; t++) {
		//randomly pick a pair of agents
		agent_pair agentPair = ran_pair_nearest(n_agents, idum, alpha, agent_matrix);
		i = agentPair.i;
		j = agentPair.j;

		//calculate original total money of the two agents
		totalM2 = agent_matrix[i] + agent_matrix[j];
		//calculate new income afer transaction
		epsilon = ran2(&idum);
		agent_matrix[i] = epsilon*totalM2;
		agent_matrix[j] = (1 - epsilon)*totalM2;
	}

	for (int i = 0; i < n_agents; i++) {
		M += agent_matrix[i];
		factor = ceil(agent_matrix[i] / dm);
		for (int j = 1; j <= factorMax; j++) {
			if (factor == j) {
				counter[j] += 1.0;
			}
		} //counting distibution of money after 1e7 transactions
	}


} // end of Metropolis sampling over agent pair


void output(int n_agents, int mcs, double total_average, int factorMax, double dm, double* total_counter)
{
	double norm = 1 / ((double)(mcs));  // divided by total number of cycles
	double Mtotal_average = total_average*norm;
	// all expectation values are per agent, divided by 1/n_agents
	ofile << setiosflags(ios::showpoint | ios::uppercase);
	ofile << setw(15) << "average money";
	ofile << setw(15) << setprecision(8) << Mtotal_average / n_agents << endl;
	for (int i = 1; i <= factorMax; i++) {
		ofile << setw(15) << setprecision(8) << i*dm << setw(15);
		ofile << setw(15) << setprecision(8) << total_counter[i] * norm << endl;
	}
	cout << "finished " << endl;
} // end output function

  /***          random pair function ran_pair()             */

agent_pair ran_pair(int n_agents, long& idum)
{
	agent_pair Agents;
	int i = 0, j = 0;
	i = (int)(ran2(&idum)*(double)n_agents);
	j = (int)(ran2(&idum)*(double)n_agents);
	while (i == j) {
		j = (int)(ran2(&idum)*(double)n_agents);
	}

	Agents.i = i;
	Agents.j = j;
	return Agents;


}

agent_pair ran_pair_nearest(int n_agents, long& idum, double alpha, double* agent_matrix)
{
	agent_pair Agents;
	int i = 0, j = 0;
	i = (int)(ran2(&idum)*(double)n_agents);
	
	vector<double> freq;

	for (int k = 0; k < n_agents; k++) {
		freq.push_back(0);
		if (k == i) {
			freq[k] = 0;
		}
		else {
			freq[k] = 1.0 / pow(abs(agent_matrix[k] - agent_matrix[i]), alpha);
		}
		
	}

	
	
	random_device seed;
	mt19937 engine(seed());
	//uniform_int_distribution<int> values(0, 499);
	discrete_distribution<> discrete(freq.begin(), freq.end());

	while (i == j) {
		j = discrete(engine);
	}
	//trans_agent1 = (int)values(engine);

	
	Agents.i = i;
	Agents.j = j;

	return Agents;


}


/*
** The function
**         ran2()
** is a long periode (> 2 x 10^18) random number generator of
** L'Ecuyer and Bays-Durham shuffle and added safeguards.
** Call with idum a negative integer to initialize; thereafter,
** do not alter idum between sucessive deviates in a
** sequence. RNMX should approximate the largest floating point value
** that is less than 1.
** The function returns a uniform deviate between 0.0 and 1.0
** (exclusive of end-point values).
*/

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double ran2(long *idum)
{
	int            j;
	long           k;
	static long    idum2 = 123456789;
	static long    iy = 0;
	static long    iv[NTAB];
	double         temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum = 1;
		else             *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB + 7; j >= 0; j--) {
			k = (*idum) / IQ1;
			*idum = IA1*(*idum - k*IQ1) - k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB)  iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1;
	*idum = IA1*(*idum - k*IQ1) - k*IR1;
	if (*idum < 0) *idum += IM1;
	k = idum2 / IQ2;
	idum2 = IA2*(idum2 - k*IQ2) - k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j = iy / NDIV;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp = AM*iy) > RNMX) return RNMX;
	else return temp;
}
#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX

// End: function ran2()


/*
* The function
*      void  **matrix()
* reserves dynamic memory for a two-dimensional matrix
* using the C++ command new . No initialization of the elements.
* Input data:
*  int row      - number of  rows
*  int col      - number of columns
*  int num_bytes- number of bytes for each
*                 element
* Returns a void  **pointer to the reserved memory location.
*/

void **matrix(int row, int col, int num_bytes)
{
	int      i, num;
	char     **pointer, *ptr;

	pointer = new(nothrow) char*[row];
	if (!pointer) {
		cout << "Exception handling: Memory allocation failed";
		cout << " for " << row << "row addresses !" << endl;
		return NULL;
	}
	i = (row * col * num_bytes) / sizeof(char);
	pointer[0] = new(nothrow) char[i];
	if (!pointer[0]) {
		cout << "Exception handling: Memory allocation failed";
		cout << " for address to " << i << " characters !" << endl;
		return NULL;
	}
	ptr = pointer[0];
	num = col * num_bytes;
	for (i = 0; i < row; i++, ptr += num) {
		pointer[i] = ptr;
	}

	return  (void **)pointer;

} // end: function void **matrix()

  /*
  * The function
  *      void free_matrix()
  * releases the memory reserved by the function matrix()
  *for the two-dimensional matrix[][]
  * Input data:
  *  void far **matr - pointer to the matrix
  */

void free_matrix(void **matr)
{

	delete[](char *) matr[0];
	delete[] matr;

} // End:  function free_matrix()