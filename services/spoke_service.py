import requests
import ast
from get_env_variables import getEnvVariables
import pandas as pd
import json

config_data, system_prompts = getEnvVariables()

def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    if params:
        return requests.get(uri, params=params)
    else:
        return requests.get(uri)

def get_context_using_spoke_api(node_value):
    with open('./data/spoke_types.json', 'r') as file_spoke_types: # save the result from api /types
        data_spoke_types = json.load(file_spoke_types)
    
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]
    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth' : config_data['depth']
    }
    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()

    if (node_context == []):
        return 'no_data', pd.DataFrame()
    
    nbr_nodes = []
    nbr_edges = []
    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]: # nodes
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        else: # edges
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)
                except:
                    try:
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = map(lambda x:"pubmedId:"+x, pmid_list)
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:
                        provenance = "SPOKE-KG"
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append((item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))
    
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])

    node_type_map = nbr_nodes_df.set_index('node_id')['node_type'].to_dict()
    node_name_map = nbr_nodes_df.set_index('node_id')['node_name'].to_dict()

    # Map source and target nodes in nbr_edges_df to their types and names
    df = nbr_edges_df
    df['source_type'] = df['source'].map(node_type_map)
    df['source_name'] = df['source'].map(node_name_map)
    df['target_type'] = df['target'].map(node_type_map)
    df['target_name'] = df['target'].map(node_name_map)
    df = df [
        [ 'source', 'source_type', 'source_name', 'edge_type', 'target', 'target_type', 'target_name', 'provenance', 'evidence'] # source and target: id
    ]

    df.loc[:, 'predicate'] = df['edge_type'].apply(lambda x: x.split('_')[0])
    df.loc[:, 'context'] = df['source_type'] + " " + df['source_name'] + " " + df['predicate'].str.lower() + " " + df['target_type'] + " " + df['target_name'] + ' and Provenance of this association is ' + df['provenance'] + '.'
    context = df['context'].str.cat(sep=' ')

    node_info = node_context[0]
    node_detail = node_value + node_info['data']['properties']['source'] + ' identifier of ' + node_info["data"]["properties"]["identifier"] + ' and Provenance of this is from ' + node_info["data"]["properties"]["source"] + "."
    context = context + node_detail
    return context, df
