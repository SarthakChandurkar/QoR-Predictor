#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <sstream>
using namespace std;

vector<string> SUBCOMMANDS = {"rwz", "rf", "rfz", "rs","b","fsto", "rw","rs -K 6","rs -K 8","rs -K 10","rs -K 12","rs -K 14","rs -K 16","rs -K 6 -N 2","rs -K 8 -N 2"
				,"rs -K 10 -N 2","rs -K 12 -N 2","rs -K 14 -N 2","rs -K 16 -N 2"};

vector<string> recipe = {"b", "rwz", "rf", "rs -K 6", "rf", "rwz", "fsto", "b", "rw", "rfz", "rfz", "rfz", "rwz", "rs -K 6 -N 2", "fsto", "b", "rw", "rfz", "rwz", "rfz", "rfz", "rfz", "fsto" };

vector<string> INTERVALEDCOM = {"b", "fsto", "rw"};
string orgCmnd()
{
	string res = "";
	for(int i=0;i<recipe.size();i++)
	{
		res = res + recipe[i] + "; ";
	}
	return res;
}
string recipeGen(vector<string> &currAneal)
{
	string res = "";
	for(int i=0;i<currAneal.size();i++)
	{
		res = res + currAneal[i] + "; ";
	}
	return res;
}
void randomize(vector<string> &currAneal)
{
	int l = rand()%currAneal.size();
	int h = rand()%currAneal.size();
	string lChange  = SUBCOMMANDS[rand()%SUBCOMMANDS.size()];
	string rChange  = SUBCOMMANDS[rand()%SUBCOMMANDS.size()];
	currAneal[l] = lChange;
	currAneal[h] = rChange;
}
string run_abc_command(const string& random_cmd, const string& search_term) {
    ostringstream command;

    command << "./abc -c \"read_lib nangate_45.lib; read_bench simple_spi_orig.bench; st; "
            << random_cmd << " map; print_stats\" | awk '/" << search_term 
            << "/ { for (i=1; i<=NF; i++) if ($i == \"" << search_term << "\") print $(i+1) }'";

    cout << "Executing command: " << command.str() << std::endl;

    FILE* pipe = popen(command.str().c_str(), "r");
    if (!pipe) return "Error";

    char buffer[128];
    string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);

    size_t pos = result.find('=');
    if (pos != string::npos) {
        return result.substr(pos + 1);
    }
    return "N/A";
}

int main() {
    srand(static_cast<unsigned int>(time(0)));
    ofstream output_file("abc_stats_anealDelay_Best.csv",ios:: app);
    if (!output_file.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return 1;
    }
    output_file << "Command,Delay,Area\n";
	string org_cmd = orgCmnd();
	org_cmd = org_cmd + "fres; ";
        cout << "Running command: " << org_cmd << std::endl;

        string final_delay = run_abc_command(org_cmd, "delay");
        string final_area = run_abc_command(org_cmd, "area");
	
        cout << "Final Delay: " << final_delay << endl;
        cout << "Final Area: " << final_area << endl;
        double fnlDly = stof(final_delay);
        double fnlAra = stof(final_area);
        double qor = fnlDly*fnlAra;
	output_file << '"' << org_cmd << "\",\"" << final_delay << "\",\"" << final_area << "\",\"" << qor << "\"\n";
        vector<string> aneal = recipe;
    for (int i = 0; i < 300; i++) {
    	vector<string> currAneal = aneal;
    	randomize(currAneal);
        string random_cmd = recipeGen(currAneal);
        random_cmd = random_cmd + "fres; ";
        cout << "Running command: " << random_cmd << std::endl;
	
        final_delay = run_abc_command(random_cmd, "delay");
        final_area = run_abc_command(random_cmd, "area");
	
        cout << "Final Delay: " << final_delay << endl;
        cout << "Final Area: " << final_area << endl;
        double rndDly = stof(final_delay);
        double rndAra = stof(final_area);
        double rndQor = rndDly*rndAra;
        if(rndDly < fnlDly)
        {
		fnlDly = rndDly;        
		output_file << '"' << random_cmd << "\",\"" << final_delay << "\",\"" << final_area << "\",\"" << rndQor << "\"\n";
	        aneal = currAneal;
        }
    }

    cout << "Stats have been saved to abc_stats_aneal_fourthBest.csv" << endl;
    return 0;
}
