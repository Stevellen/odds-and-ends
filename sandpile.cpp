#include <stdexcept>
#include <iostream>
#include <windows.h>
#include <assert.h>
#include <vector>
#include <string>
#include <map>

#define ASSERT(condition, message) if (!(condition)) { std::cerr << message << std::endl; assert(condition); }

#define CLEAR_FRAME() { \
    static const HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE); \
    COORD topLeft = { 0, 0 }; \
    SetConsoleCursorPosition(hOut, topLeft); \
}

class Sandpile {
    private: // private data members
        static const std::map<int, std::string> pm_charmap;
    public: // public data members
        int m_length, m_width;
        std::vector<std::vector<int>> m_value;

    public: // function definitions
        // Constructors
        Sandpile() : m_length(100), m_width(100), m_value(100, std::vector<int>(100, 0)) {}
        Sandpile(const int& length, const int& width) : m_length(length), m_width(width), m_value(length, std::vector<int>(width, 0)) {}
        Sandpile(std::vector<std::vector<int>> value) : m_length(value.size()), m_width(value[0].size()), m_value(value) {}

        /* STATIC DEFINITIONS */
        static std::map<int, std::string> createMap() {
            std::map<int, std::string> m;
            m[0] = ".";
            m[1] = "o";
            m[2] = "O";
            m[3] = "0";
            return m;
        }
        
        /* SOME BASIC TEMPLATE PATTERNS FOR ANIMATION GENERATION */
        static Sandpile makeDrop(int length, int width) {
            std::vector<std::vector<int>> grid(length, std::vector<int>(width, 0));
            grid[length / 2][width / 2] = 1;
            return Sandpile(grid);
        }

        static Sandpile makePlus(int length, int width) {
            std::vector<std::vector<int>> grid(length, std::vector<int>(width, 0));
            for (int ii = 0; ii < length; ++ii) {
                for (int jj = 0; jj < width; ++jj) {
                    if (ii == length/2 || jj == width/2) {
                        grid[ii][jj] = 1;
                    }
                }
            }
            return Sandpile(grid);
        }

        static Sandpile makeX(int length, int width) {
            ASSERT(length == width, "Cannot make a cross on a non-square grid!!");
            std::vector<std::vector<int>> grid(length, std::vector<int>(width, 0));
            for (int ii = 0; ii < length; ++ii) {
                for (int jj = 0; jj < width; ++jj) {
                    if (ii == jj || ii+jj == length-1) grid[ii][jj] = 1;
                }
            }
            return Sandpile(grid);
        }

        static void animate(int length, int width, int iterations, std::string pattern = "drop") {
            Sandpile grid(length, width);
            Sandpile s;


            if (pattern == "drop") {
                s = Sandpile::makeDrop(length, width);
            }
            else if (pattern == "plus") {
                s = Sandpile::makePlus(length, width);
            }
            else if (pattern == "x") {
                s = Sandpile::makeX(length, width);
            }
            else {
                throw std::runtime_error("Invalid pattern!");
            }
            CLEAR_FRAME()
            // update and print for each iteration
            for (int ii = 0; ii < iterations; ++ii) {
                grid.print();
                grid = grid + s;
                Sleep(25);
                if (ii != iterations-1) CLEAR_FRAME();
            }
            std::cout << std::endl;
        }

        // Other definitions
        void print() {// prints current state to console
            for(std::vector<int> xs : m_value) {
                for(int x : xs) {
                    std::cout << pm_charmap.find(x)->second << " ";
                }
                std::cout << std::endl;
            }
        }

        Sandpile operator+(const Sandpile& other) const {
            bool hasPile = false;
            ASSERT(other.m_length == m_length && other.m_width == m_width, "Sandpiles of different sizes cannot be combined!");
            
            // create container for sum
            std::vector<std::vector<int>> sum(m_length, std::vector<int>(m_width, 0));

            // iterate over vectors and add up values
            for(int ii = 0; ii < m_length; ++ii) {
                for(int jj = 0; jj < m_width; jj++) {
                    sum[ii][jj] = other.m_value[ii][jj] + m_value[ii][jj];
                    if(sum[ii][jj] >= 4) hasPile = true;
                }
            }
            
            while(hasPile) { // while there are piles, iterate and topple
                hasPile = false;
                // now perform update on sum for max height
                for(int ii = 0; ii < m_length; ++ii) {
                    for(int jj = 0; jj < m_width; jj++) {
                        if(sum[ii][jj] >= 4) {
                            sum[ii][jj] -= 4;
                            if(ii-1 >= 0) { // check index of cell above
                                sum[ii-1][jj] += 1;
                                if (sum[ii-1][jj] >= 4) hasPile = true;
                            }
                            if(ii+1 <= m_length-1) { // check index of cell below
                                sum[ii+1][jj] += 1;
                                if (sum[ii+1][jj] >= 4) hasPile = true;
                            }
                            
                            if(jj-1 >= 0) { // check index of cell to the left
                                sum[ii][jj-1] += 1;
                                if (sum[ii][jj-1] >= 4) hasPile = true;
                            }
                            
                            if(jj+1 <= m_width-1) { // check index of cell to the right
                                sum[ii][jj+1] += 1;
                                if (sum[ii][jj+1] >= 4) hasPile = true;
                            }
                        }
                    }
                }
            }

            return Sandpile(sum);
        }

        // TODO: figure out how to make this work
        // Sandpile operator+=(const Sandpile& other) const {
        //     Sandpile new_pile = (*this) + other;
        //     m_value = new_pile.m_value;
        //     return *this;
        // }

};
const std::map<int, std::string> Sandpile::pm_charmap = Sandpile::createMap();


int main(int argc, const char** argv) {
    // Basic animation demo
    Sandpile::animate(64, 64, 250, "x");
    return 0;
}