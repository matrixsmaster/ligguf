#pragma once

#include <string>
#include <vector>
#include "lil_gguf.h"

#define VDB_MAX_RETRIEVAL 3
#define VDB_LIKENESS_THRESHOLD 0.75

struct vdb_entry {
    uint32_t timestamp;
    std::vector<gguf_half> vec;
    std::string context_before, user_text, context_after;
};

typedef std::vector<vdb_entry> vecdb;

void vdb_set_model(std::string model_fn);
void vdb_set_names(std::string ai_name, std::string user_name);
void vdb_create(std::string log_text);
vecdb vdb_query(std::string intext);
std::string vdb_to_prompt(const vecdb &mem);
void vdb_log_turn(std::string &log, std::string user, std::string ai);
