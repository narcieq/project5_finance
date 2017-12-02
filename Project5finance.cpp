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