import os



import config

from reflection_engine import Reflec

from strategies.base_searcher import BaseSearcher

from state_manager import StateManager





class SingleSearcher(BaseSearcher):

    def __init__(self, llm_agent):

        super().__init__(llm_agent)

        self.reflector = self._build_reflector()

        self.initial_lessons_ready = False

        self.cached_lessons_text = self.current_strategic_experience



    def _build_reflector(self):

        try:

            with open(config.REFLECT_PROMPT_FILE, "r", encoding="utf-8") as reflection_file:

                reflection_prompt = reflection_file.read()

        except OSError:

            return None

        return Reflec(prompt_template=reflection_prompt)



    def _prepare_single_search_lessons(

        self,

        initial_state: StateManager,

        session_folder: str,

        iteration: int,

    ) -> str:

        if self.reflector is None:

            return self.current_strategic_experience

        if not getattr(config, "USE_REFLECTION", True):

            return self.current_strategic_experience



        lessons = self.reflector.execute_hierarchical_reflection(

            initial_state,

            "SingleSearch presearch warmup",

            session_folder,

            iteration,

        )

        return lessons.strip() if lessons.strip() else self.current_strategic_experience



    def _save_lessons_file(self, session_folder: str, lessons_text: str) -> None:

        lessons_file_path = os.path.join(session_folder, "lessons.md")

        with open(lessons_file_path, "w", encoding="utf-8") as lessons_file:

            lessons_file.write(lessons_text)



    def run_search(self, initial_state: StateManager, session_folder: str, iteration: int) -> dict:

        """

        Baseline strategy: No tree search. Directly asks the LLM for 1 action.

        """

        feasible_actions = initial_state.get_feasible_actions()



        if not feasible_actions:

            return None



        if not self.initial_lessons_ready:

            self.cached_lessons_text = self._prepare_single_search_lessons(

                initial_state,

                session_folder,

                iteration,

            )

            if not self.cached_lessons_text.strip():

                self.cached_lessons_text = self.current_strategic_experience

            self._save_lessons_file(session_folder, self.cached_lessons_text)

            self.initial_lessons_ready = True



        if len(feasible_actions) == 1:

            forced_action = feasible_actions[0]

            print(">>> [SingleSearch] Short-circuit: only 1 feasible action. Skipping LLM call.")

            return forced_action



        print(f">>> [SingleSearch] Evaluating {len(feasible_actions)} immediate actions...")



        decision = self.llm_agent.get_action(

            state=initial_state,

            feasible_actions=feasible_actions,

            strategic_experience=self.cached_lessons_text,

            session_folder=session_folder,

            iteration=iteration,

        )



        return decision

