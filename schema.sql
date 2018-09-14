--
-- PostgreSQL database dump
--

-- Dumped from database version 10.3 (Debian 10.3-1.pgdg90+1)
-- Dumped by pg_dump version 10.4

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: DATABASE postgres; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE postgres IS 'default administrative connection database';


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;


--
-- Name: EXTENSION pg_trgm; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pg_trgm IS 'text similarity measurement and index searching based on trigrams';


SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: match; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.match (
    start integer NOT NULL,
    "end" integer NOT NULL,
    weight double precision NOT NULL,
    CONSTRAINT max_check CHECK ((weight <= (1)::double precision)),
    CONSTRAINT min_check CHECK ((weight > (0)::double precision))
);


ALTER TABLE public.match OWNER TO postgres;

--
-- Name: query_edge; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.query_edge (
    start integer NOT NULL,
    "end" integer NOT NULL
);


ALTER TABLE public.query_edge OWNER TO postgres;

--
-- Name: query_node; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.query_node (
    id integer NOT NULL,
    label character varying,
    type integer
);


ALTER TABLE public.query_node OWNER TO postgres;

--
-- Name: query_node_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.query_node_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.query_node_id_seq OWNER TO postgres;

--
-- Name: query_node_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.query_node_id_seq OWNED BY public.query_node.id;


--
-- Name: target_edge; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.target_edge (
    start integer NOT NULL,
    "end" integer NOT NULL
);


ALTER TABLE public.target_edge OWNER TO postgres;

--
-- Name: target_node; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.target_node (
    id integer NOT NULL,
    label character varying,
    mbid character varying,
    type integer
);


ALTER TABLE public.target_node OWNER TO postgres;

--
-- Name: target_node_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.target_node_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.target_node_id_seq OWNER TO postgres;

--
-- Name: target_node_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.target_node_id_seq OWNED BY public.target_node.id;


--
-- Name: query_node id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_node ALTER COLUMN id SET DEFAULT nextval('public.query_node_id_seq'::regclass);


--
-- Name: target_node id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.target_node ALTER COLUMN id SET DEFAULT nextval('public.target_node_id_seq'::regclass);


--
-- Name: match match_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.match
    ADD CONSTRAINT match_pkey PRIMARY KEY (start, "end");


--
-- Name: query_edge query_edge_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_edge
    ADD CONSTRAINT query_edge_pkey PRIMARY KEY (start, "end");


--
-- Name: query_node query_node_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_node
    ADD CONSTRAINT query_node_pkey PRIMARY KEY (id);


--
-- Name: target_edge target_edge_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.target_edge
    ADD CONSTRAINT target_edge_pkey PRIMARY KEY (start, "end");


--
-- Name: target_node target_node_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.target_node
    ADD CONSTRAINT target_node_pkey PRIMARY KEY (id);


--
-- Name: match match_end_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.match
    ADD CONSTRAINT match_end_fkey FOREIGN KEY ("end") REFERENCES public.target_node(id);


--
-- Name: match match_start_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.match
    ADD CONSTRAINT match_start_fkey FOREIGN KEY (start) REFERENCES public.query_node(id);


--
-- Name: query_edge query_edge_end_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_edge
    ADD CONSTRAINT query_edge_end_fkey FOREIGN KEY ("end") REFERENCES public.query_node(id);


--
-- Name: query_edge query_edge_start_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.query_edge
    ADD CONSTRAINT query_edge_start_fkey FOREIGN KEY (start) REFERENCES public.query_node(id);


--
-- Name: target_edge target_edge_end_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.target_edge
    ADD CONSTRAINT target_edge_end_fkey FOREIGN KEY ("end") REFERENCES public.target_node(id);


--
-- Name: target_edge target_edge_start_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.target_edge
    ADD CONSTRAINT target_edge_start_fkey FOREIGN KEY (start) REFERENCES public.target_node(id);


--
-- PostgreSQL database dump complete
--

