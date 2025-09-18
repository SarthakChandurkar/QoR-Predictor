#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <sstream>
using namespace std;

//vector<string> SUBCOMMANDS = {"rw", "rf", "rfz", "rs -K 6", "rs -K 8", "rs -K 10", "rs -K 12", "rs -K 16"};
vector<string> SUBCOMMANDS = {"rwz", "rf", "rfz", "rs"};

vector<string> INTERVALEDCOM = {"b", "fsto", "rw"};
vector<string> subV = {"6","8","12","14","16"};
string generate_random_command() {
    ostringstream random_command;

    random_command << INTERVALEDCOM[0] << "; "; // Start with b
    int cnt = 0;
    for (int i = 1; i < 18; i++) {
        if (i % 6 == 0) {
            // Add intervaled commands after every 3 commands
            random_command << INTERVALEDCOM[1] << "; "; // ftso
            random_command << INTERVALEDCOM[0] << "; "; // b
            random_command << INTERVALEDCOM[2] << "; "; //rw
        }
        else{
            int rand_index = rand() % SUBCOMMANDS.size();
            if(rand_index == 3)
            {
                int index = cnt/2;
                if(index >= subV.size())
                {
                    i--;
                    continue;
                }
                if(cnt&1)
                {
                    string rs = SUBCOMMANDS[rand_index] + " -K " +  subV[index] + " -N 2";
                    random_command << rs << "; ";
                }
                else
                {
                    string rs = SUBCOMMANDS[rand_index] + " -K " +  subV[index];
                    random_command << rs << "; ";
                }

                cnt++;
            }
            else
                random_command << SUBCOMMANDS[rand_index] << "; ";
        }
    }
    random_command << "fsto; "; // fraig store
    random_command << "fres; "; // End with fres
    return random_command.str();
}

string run_abc_command(const string& random_cmd, const string& search_term) {
    ostringstream command;

    command << "./abc -c \"read_lib nangate_45.lib; read_bench tv80_orig.bench; st; "
            << random_cmd << " map; print_stats\" | awk '/" << search_term 
            << "/ { for (i=1; i<=NF; i++) if ($i == \"" << search_term << "\") print $(i+1) }'";

    //cout << "Executing command: " << command.str() << std::endl;

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
    ofstream output_file("abc_stats_mid.csv");
    if (!output_file.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return 1;
    }
    output_file << "Command,Delay,Area\n";

    for (int i = 0; i < 100; i++) {
        string random_cmd = generate_random_command();
        cout << "Running command: " << random_cmd << std::endl;

        string final_delay = run_abc_command(random_cmd, "delay");
        string final_area = run_abc_command(random_cmd, "area");

        cout << "Final Delay: " << final_delay << endl;
        cout << "Final Area: " << final_area << endl;
        output_file << '"' << random_cmd << "\",\"" << final_delay << "\",\"" << final_area << "\"\n";
    }

    cout << "Stats have been saved to abc_stats_mid.csv" << endl;
    return 0;
}
