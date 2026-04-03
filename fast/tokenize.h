#pragma once

#include <vector>
#include <string>

std::vector<int> tokenize(const char* str, bool bos, bool eos);
std::string tok_to_str(int tok);
